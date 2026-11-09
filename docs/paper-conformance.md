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
- The quaternion is normalized before R (Eq. 10), and its gradient goes through the normalization,
  dL/dq = (dL/dn − n(n·dL/dn))/|q|: the gradient of the stored quaternion, as the reference's
  PyTorch `normalize` gives it by autograd.
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
- The near plane at z = 0.2 (a parameter of `projectSplat`).
- The Jacobian evaluated with x/z and y/z clamped to 1.3 tan(fov/2). The reference assumes a
  centred principal point; `lib/splat_math.zgal` extends the image by 0.15 of its size on each side
  instead ([(−cx − 0.15 W)/fx, (1.15 W − cx)/fx]), which is the same for a centred one.
- The SH basis's constants, signs and coefficient order (`computeColorFromSH`).

Training, from the reference repository's `arguments/__init__.py`, `train.py` and
`gaussian_model.py` at its latest release (182afc9, October 2024; splat_train's defaults,
[architecture.md](architecture.md#splat_train)):

- Learning rates: position 1.6e-4 decaying log-linearly to 1.6e-6 over 30,000 iterations
  (`get_expon_lr_func`; its delay multiplier has no effect with the default 0 delay steps), times
  the scene extent; SH DC 2.5e-3, the other bands 1/20 of that; opacity 0.025 (the 2023 release's,
  from the paper's time, was 0.05, until the August 2024 release, 2130164; the others haven't
  changed since July 2023); scale 5e-3; rotation 1e-3.
- Adam: β = (0.9, 0.999), ε = 1e-15, one state per parameter group, with bias correction.
- The SH degree: one band added every 1000 iterations up to the scene's, the iteration counted
  from 1 (iteration 1000 renders with degree 1).
- The cameras: one per iteration, popped at random from a stack refilled when empty (each epoch
  sees every training view once). splat_train picks with `pcgHash` of the seed and the iteration
  instead of Python's `random`, the same order on every platform.
- A black background (`white_background` off) and no random background.
- Evaluation: PSNR on the render clamped to [0, 1], averaged over the test views.
- `percent_dense` 0.01 (the clone/split size threshold, relative to the scene extent).
- Pruning below opacity 0.005, and above 20 px on screen after the first opacity reset (as the
  code is written: it never fires, [departure 12](#departures)).
- The loss: (1 − λ)·L1 + λ·D-SSIM with λ = 0.2 (splat_train's `--lambda-dssim`; 0 trains on L1
  alone), D-SSIM = 1 − SSIM as `loss_utils.py`'s `ssim` computes it: per channel, an 11×11
  Gaussian window with σ 1.5, C1 = 0.01², C2 = 0.03², on the rasterizer's output before any
  clamp, and `conv2d` with padding 5, so the window is cut off at the edges and the zeros count
  toward the means. The gradient is exact for that function rather than autograd's, checked
  against finite differences.

## Departures

1. **Sort key**: 32 bits (tile, then the depth quantized by its float bits, close to log spaced:
   [architecture.md](architecture.md#splat-math)) instead of 64, with ties in splat-index order.
   Fallback if depth quantization shows: two sorts.
2. **Capacity**: a fixed splat capacity (the required `--max-splats`), compacted by scan, where the
   paper's count is unbounded. ZGAL recreates a buffer, losing its contents, when its per-frame
   size changes, so only scratch buffers grow.
3. **Gradient accumulation**: per-tile shared-memory accumulation and a per-splat gather instead of
   global float atomics. The devices have them (MoltenVK and lavapipe report buffer float
   atomic-add), but global atomics contend on popular splats and are a CAS loop on the web, and
   their summation order varies from run to run, so the gradients wouldn't be deterministic.
4. **Densification gradient**: the norm of the 2D-mean gradient, as in the reference code, and in
   its units: per normalized device coordinate, |(W/2·∂L/∂x, H/2·∂L/∂y)| with the pixel gradient
   (`densifyNorm`), so the threshold τ = 0.0002 applies unchanged. The paper says only
   "view-space positional gradients".
5. **No whole-tile early exit**: ZGAL barriers can't sit in a loop that depends on shared memory.
   Each pixel still stops blending on its own.
6. **No low-resolution warm-up.**
7. **Tile rectangle**: [⌊(m − r)/16⌋, ⌈(m + r)/16⌉) per axis, the tiles meeting the square
   (m − r, m + r), symmetric and covering every pixel centre in it. The reference's `getRect`, in
   this repository's +0.5 pixel-centre convention, is [⌊(m − 0.5 − r)/16⌋, ⌊(m − 0.5 + r + 15)/16⌋):
   it drops tile k when m + r ∈ (16k, 16k + 1.5), though for m + r ≥ 16k + 0.5 that tile holds
   pixel centre 16k + 0.5 inside the square, and adds tile k − 1 when m − r ∈ [16k, 16k + 0.5).
   It shows in images: at 3σ, α = opacity·e^−4.5 ≈ 0.011·opacity, above the 1/255 cutoff, so
   pixels in those tiles blend differently from a reference render.

8. **Alpha clamp's gradient**: where α = min(0.99, opacity·G) clamps, the backward rasterize
   gives the opacity, mean and conic a gradient of 0 (the exact derivative). The reference code
   passes the gradient through the clamp as if it didn't hold. It matters only for splats with
   opacity·G > 0.99, rare after the sigmoid.
9. **Conic gradient convention**: the conic's b gradient counts both off-diagonal entries;
   the reference's `dL_dconic2D.y` is half of it ([building-and-testing.md](building-and-testing.md#closed-form-oracles)).
   The preprocess chain must take it as it is.
10. **The Jacobian clamp's gradient**: where x/z (or y/z) is clamped to the frustum guard, J
    uses the clamped ratio, and its gradient passes nothing to x (or y) and differentiates the
    clamped value through z alone, the exact derivative. The reference code also zeroes x's
    gradient there, but its z derivative uses the unclamped formula, as if the clamped x were
    fixed rather than the ratio times z: twice the exact derivative of J's third column. Only
    splats far outside the view (past 1.3·tan(fov/2)) are affected.

11. **Skipped iterations**: a frame whose tile entries overflow the current capacity loses
    splats, so its gradients are incomplete; Adam skips it (the iteration is counted and reported,
    and the capacity grows for the next). The reference allocates per frame and never truncates.
    Adam's step count still advances on a skipped frame, so its bias corrections run one step
    ahead of the reference's per skip, which is negligible past the first iterations. A
    densifying frame doesn't step Adam and isn't counted, as in the reference (its step finds no
    gradients on the new tensors, so it does nothing): the count is the iterations less the
    densifications.
12. **Screen-space prune, a choice** (`maxScreenSize`): after the first opacity reset, a splat
    whose largest radius since the last densification passed `maxScreenSize` pixels is pruned, as
    the reference code means to with its `max_screen_size` of 20; 0 prunes none. The reference's
    never fires: `densification_postfix` zeroes `max_radii2D` before `prune` reads it, so its
    published results are 0's (large background and near-camera splats survive), while 20 is what
    it was written to do. Neither is hidden as a default: the app chooses (splat_train's
    `--max-screen-size` is required). A split's halves aren't checked by radius (their radii
    aren't known; the reference's world-space check at the halves' scale applies to them, as it
    does to every splat after the first reset, with either choice).
13. **Densification order**: the splats are compacted in place by a scan (survivors in order, a
    split's first half in its parent's place), then the additions (clones' copies and splits'
    second halves) in their parents' order. The reference appends the clones, then the halves
    after it removes the split parents, and prunes last; the set of splats is the same.
14. **Capacity overflow**: an addition past `--max-splats` is dropped and counted: the clone isn't
    made, the split isn't done (the splat is kept, with its Adam moments). The reference has no
    cap.
15. **Split sampling**: the halves' offsets are standard normals from a PCG hash of the
    densification's seed, the splat's index and the half (Box-Muller), not torch's generator, so
    runs are deterministic on every backend but don't match the reference's samples.
16. **Opacity reset and Adam**: a resetting frame steps Adam, opacity included, and then resets
    the opacity (and zeroes its moments). The reference resets first, and its step then skips
    the opacity (the new tensor has no gradient) and leaves the opacity's step count behind the
    others'. A reset on a densifying frame steps nothing in either, and with the default schedule
    the periodic resets (multiples of 3000) all do. The others differ: the white-background reset
    at `--densify-from` (not a densifying frame, which needs t past it), any reset at or before
    `--densify-from`, and every reset when `--reset-interval` isn't a multiple of
    `--densify-interval`. On those the opacity takes one more step before the reset's clamp to
    0.01, and its bias corrections count that frame.
17. **The last iteration**: it neither densifies nor resets the opacity (the reference scores and
    saves iteration t before its densification and reset, so what those do on the last iteration
    is never scored or written), but it steps Adam, which the reference's last iteration doesn't
    (`if iteration < opt.iterations`): the scored splats have had one more step, one of N.
18. **The viewer has no early stop** (`lib/view_graph.zgal`): it blends back to front, so it can't
    know where a pixel's front-to-back transmittance falls below `transmittanceMin` (1e-4), and it
    blends every splat whose alpha passes. Rasterize, as the reference, leaves out the splat that
    would take T below it and every splat after. It adds no visible artifact. The splats behind
    that one are seen through T < 1e-4 together: under 0.03/255. The one larger term is the
    crossing splat itself, which Rasterize leaves out (the background shows through its T) and
    the viewer blends. T before it is below `transmittanceMin`/(1 − α) ≤ 0.01 (α ≤ `alphaMax`),
    so the pixel changes by at most T·|colour − background|: up to 2.6/255 per unit of colour
    difference in a channel (2.6/255 for colours in [0, 1]; `shColour` has no upper clamp, so a
    brighter splat scales it, about 5/255 at colour 2), and over 1/255 only where the crossing
    splat is almost opaque (α > 0.975). That is a single-pixel difference of a few levels at most,
    not an edge or a band. On S3, Rasterize without the stop changes 9 of 16384
    pixels (camera 0) and 13 of 15360 (camera 24), each by 1/255.
19. **The viewer blends a splat only inside its 3σ square**: SplatQuad's quad is the square of the
    splat's radius around its mean, the radius being 3σ along the larger axis, rounded up to
    pixels. Rasterize evaluates every pixel of the tiles that square meets (departure 7), so it
    also blends beyond the square wherever α passes `alphaMin`. It adds no visible artifact.
    Beyond the square α ≤ opacity·e^−4.5 ≈ 0.011·opacity, so the viewer leaves out at most
    0.011·T·|colour − behind| ≤ 2.8/255 a splat per unit of colour difference (it scales with a
    colour above 1, as in departure 18), and nothing at an opacity below 0.353 (e^4.5/255; α under
    `alphaMin`: why the viewer's closed-form fixtures keep o < 0.35). The reference's own
    footprint is cut at its tile rectangle, a step of the same size at tile edges. On S3 (opacity
    0.3-0.95), clipping Rasterize to the square changes no pixel.

## How correctness is checked

Finite differences and re-fitting check the code against itself, so they can't catch a forward
pass that is wrong but consistent. Five checks together cover it:

- closed-form single-splat oracles for renders and gradients (`tests/oracle/`);
- reprojecting every COLMAP point onto its recorded observations (the camera conventions): the
  recomputed mean reprojection errors match COLMAP's recorded ones to 1e-6 px
  (`tools/tests/test_colmap_reprojection.py`);
- GPU finite differences of the whole forward pass, on tiny synthetic scenes: central differences
  with h = 1e-3 relative (1e-4 absolute floor) on a smooth L2 test loss, passing when
  |analytic − FD| ≤ 2e-2·max(|analytic|, |FD|) + 1e-5 (plus, for the quaternion, 2e-3 of its
  largest gradient: the float noise of summing the pixels' changes), skipping parameters with
  negligible gradient, for the projection's fields and every parameter group and SH band;
- the SSIM kernels (`programs/ssim_test`): the SSIM against an independent float64 numpy SSIM on
  an identical, a constant and a random non-square pair, every pixel's on the constant pair
  against the closed form, zero gradients on the identical pair and the random pair's gradients
  against GPU finite differences; then training's loss, 0.8·L1 + 0.2·D-SSIM, through the
  backward pass by GPU finite differences of the projection's colour (grad_test's `--dssim`);
- re-fitting a synthetic scene of known Gaussians, then a PSNR report (no target) on the cs5330
  scene.
