# Paper conformance

The specification is Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
(SIGGRAPH 2023). Values the paper doesn't give come from its reference code
(graphdeco-inria/gaussian-splatting, `arguments/__init__.py`). This page records why the old code
was replaced, and where the rewrite follows the paper or knowingly departs from it. Add any new
departure here.

## Why the old code was replaced

The pre-ZGAL code (in history before this rewrite) was written against a wrapper API that no
longer exists, never trained in the default build (training needed `-DBACKPROP`), and had bugs at
every stage. Its camera conventions were found by trial and error ("Fix inverted view matrix",
"Undo view transpose"). It was rewritten rather than patched; the main faults, by stage:

- **Data loading**: the COLMAP reader dropped every line of 21 tokens or more (every point with 7
  or more observations) and mis-parsed images.txt's two-line records; quaternions were read xyzw
  where COLMAP writes w first; the view was `inverse(R)` although COLMAP's (R, t) is already world
  to camera, patched with sign flips for an OpenGL (−z forward) projection; SIMPLE_RADIAL
  distortion was ignored and cameras were assumed square.
- **Preprocess**: `mat3(view)` where Eq. 5 needs the transposed rotation; no +0.3 low-pass term; a
  precedence bug in the eigenvalue midpoint (`0.5*a + c`); a 10σ radius instead of 3σ; the near
  test in NDC; negative extents cast to uint, so off-screen splats were never culled; the camera
  centre taken as `view[3].xyz` instead of −Rᵀt; SH colour without the +0.5 offset, on a 0–255
  scale.
- **Tiles and sort**: a fixed 4×4 grid of tiles instead of 16×16-pixel tiles with one global sort;
  a `uint[16]` count buffer read with std140's 16-byte stride (tiles 4 and up out of bounds);
  duplicated and dropped entries; `groupMemoryBarrier()` where `barrier()` was needed; a depth key
  undefined for negative depths; a radix sort assuming 64-wide subgroups (Apple's are 32), with
  rasterize reading the buffer holding 3 of its 4 passes.
- **Rasterize**: a format mismatch on the gradient cache; the last contributor stored as a half
  float (wrong above 2048); none of Appendix C's thresholds.
- **Backward pass**: wrong tile bounds; the mean gradient multiplied by dG/dΔx twice; an L2-like
  loss on 0–255 instead of (1−λ)·L1 + λ·D-SSIM; gradients never zeroed between iterations; float
  atomics without the device feature; the 2D-to-3D mean chain through the projection alone, and
  the covariance chain repeating the forward W bug (Appendix A); SH degree 3 writing coefficients
  0–5 instead of 10–15.
- **Optimizer and densification**: plain SGD at a tenth of Adam's rates; sigmoid re-applied to the
  stored opacity every step; scale without an activation (could go negative); clone, split and
  pruning unlike Sec. 5.2, run every 10 epochs; an "iteration" meaning an epoch over all images;
  buffers recreated while frames were in flight. The splat count blew up (dumps grew from 2 MB to
  138 MB).

## Where the rewrite follows the paper

- The representation: position, log-scale, quaternion, opacity logit, SH degree 3.
- EWA projection (Eq. 5, Σ' = J W Σ Wᵀ Jᵀ) with +0.3 on the diagonal, a 3σ radius and a near-plane
  guard.
- 16×16 tiles, one global sort, per-tile ranges from comparing neighbouring keys.
- Front-to-back blending with Appendix C's thresholds: skip α < 1/255, clamp α ≤ 0.99, stop before
  T < 1e-4.
- The backward pass: a reverse traversal rebuilding T from the last contributor and the final T,
  and Appendix A's gradients (M = RS, Eq. 11).
- Adam, with sigmoid (opacity), exp (scale) and a normalized quaternion.
- The loss (1−λ)·L1 + λ·D-SSIM with λ = 0.2.
- One image per iteration, an SH band added every 1000 iterations, exponential decay of the
  position learning rate only.
- Densification every 100 iterations from 500 to 15k, at a 2D-mean gradient threshold τ = 0.0002:
  clone small splats, split large ones (new positions sampled from the Gaussian, scale divided by
  1.6); opacity reset every 3000 iterations; pruning.
- Every 8th image held out for testing.

## Values from the reference code

Rendering, from diff-gaussian-rasterization's `forward.cu` (the closed-form oracles in
`tests/oracle/` use them):

- +0.3 on the 2D covariance's diagonal (the low-pass filter).
- SH colour plus 0.5, clamped at 0 from below.
- The radius ceil(3·√λ), with λ = mid + √max(mid² − det, 0.1): the 0.1 floor makes a
  near-isotropic splat's radius up to a pixel larger than the exact eigenvalue gives.

Training, to be checked against the reference repository when training lands:

- Learning rates: position 1.6e-4 decaying to 1.6e-6, scaled by the scene extent; SH DC 2.5e-3,
  the other bands 1/20 of that; opacity 0.05; scale 5e-3; rotation 1e-3.
- `percent_dense` 0.01 (the clone/split size threshold, relative to the scene extent).
- Pruning below opacity 0.005, and above 20 px on screen after the first opacity reset.

## Departures

1. **Sort key**: 32 bits (tile, then quantized log depth) instead of 64. Fallback if depth
   quantization shows: two sorts.
2. **Capacity**: a fixed splat capacity (the required `--max-splats`), compacted by scan, where the
   paper's count is unbounded. ZGAL recreates a buffer, losing its contents, when its per-frame
   size changes, so only scratch buffers grow.
3. **Gradient accumulation**: per-tile shared-memory accumulation and a per-splat gather instead of
   global float atomics. The devices have them (MoltenVK and lavapipe report buffer float
   atomic-add), but global atomics contend on popular splats and are a CAS loop on the web, and
   their summation order varies from run to run, so the gradients wouldn't be deterministic.
4. **Densification gradient**: the norm of the 2D-mean gradient, as in the reference code.
5. **No whole-tile early exit**: ZGAL barriers can't sit in a loop that depends on shared memory.
   Each pixel still stops blending on its own.
6. **No low-resolution warm-up.**

## How correctness is checked

Finite differences and re-fitting check the code against itself, so they can't catch a forward
pass that is wrong but consistent. Four checks together cover it:

- closed-form single-splat oracles for renders and gradients (`tests/oracle/`);
- reprojecting every COLMAP point onto its recorded observations (the camera conventions): the
  recomputed mean reprojection errors match COLMAP's recorded ones to 1e-6 px
  (`tools/tests/test_colmap_reprojection.py`);
- GPU finite differences of the whole forward pass, on tiny synthetic scenes: central differences
  with h = 1e-3 relative (1e-4 absolute floor) on a smooth L2 test loss, passing when
  |analytic − FD| ≤ 2e-2·max(|analytic|, |FD|) + 1e-5, skipping parameters with negligible gradient;
- re-fitting a synthetic scene of known Gaussians, then a PSNR report (no target) on the cs5330
  scene.
