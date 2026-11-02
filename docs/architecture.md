# Architecture

## Layout

The planned layout, filled in as the M1 and M2 tasks land.

- `programs/<name>/`: one ZGAL program each (`<name>.zgal` with a `main`, plus its own `.cpp` and
  `.ts` files, `assets/` and optional `test.sh` / `test-web.sh`). The Makefile finds them; adding
  one needs no Makefile change ([building-and-testing.md](building-and-testing.md)).
  - `splat_render`: renders a camera of a scene in a window, or headless with a PNG capture
    ([below](#splat_render)).
  - `tile_test` and `render_test`: the render graph's GPU tests against the CPU and the
    closed-form oracles ([below](#render-graph)). render_test's checks are in `render_check.h`,
    shared by the native program and `web-check/render_test_check.cpp`, which checks a web run's
    readbacks ([web.md](web.md#tests)).
  - `grad_test`: the training graph's gradients against the closed-form oracle and GPU finite
    differences ([below](#train-graph)); its checks are in `grad_check.h`, shared with
    `web-check/grad_test_check.cpp` as render_test's are.
- `lib/`: ZGAL modules the programs share, imported by relative path:
  `(import "../../lib/<module>.zgal")`. A module's name is a C++ namespace (ZGAL compiles its
  functions to C++ too), so it uses underscores, not hyphens.
  - `splat_math.zgal`: the structs, the per-splat preprocess (covariance, EWA projection, conic,
    radius, tile rectangle, SH colour) and the sort keys ([below](#splat-math)).
  - `render_graph.zgal`: `SplatGraph`, the render graph's preprocess, tile entries, sort, tile
    ranges and rasterizer, and the window's present ([below](#render-graph)); it imports
    `splat_math.zgal`.
  - `train_graph.zgal`: `TrainGraph`, the render graph's forward pass followed by the loss and the
    backward rasterize, giving each splat's gradients with respect to its projection
    ([below](#train-graph)); it imports `render_graph.zgal`.
- `tools/`: offline Python (numpy, scipy, Pillow): the converter from COLMAP text and INRIA PLY files to the
  programs' inputs, the synthetic scene generator, and `scene_args.py`, which prints
  splat_render's arguments for a scene and a camera.
- `tests/`: C++ tests (`tests/cpp/`), closed-form oracles (`tests/oracle/`), Python tests and test
  data.
- `colmap/`: the one real scene: `colmap_exports3`, a COLMAP text export (one SIMPLE_RADIAL
  camera, 46 images, sparse points; CRLF line endings), and its images, `cs5330_dense_resized`
  (748×748 JPEGs, named as in `images.txt`), committed directly (6.2 MB, no LFS).
- `external/vulkan_test`: the wrapper and ZGAL.

## Pipeline

The planned pipeline (M1 renders, M2 trains). Steps 1 to 5 are `lib/render_graph.zgal`
([below](#render-graph)), and steps 6 to 9 are `lib/train_graph.zgal` ([below](#train-graph)),
with step 9's math in `lib/splat_backward.zgal`.
The rest of training is to come.

Data is converted offline: the programs read std430 `buffer-file` data and a `manifest.json`, the
same on native and the web, so no image or COLMAP parsing library is linked. The training images
are undistorted and packed by the converter into an RGBA8 buffer-file (one `uint` per pixel), not
bound as images: ZGAL binds only fixed `image-file` lists as arrays, and `import-images` is native
only.

Per frame, rendering follows the paper's tile-based rasterizer:

1. **Preprocess** (per splat): 3D covariance from scale and rotation, EWA projection to a 2D conic
   (plus 0.3 on the diagonal), a 3σ radius, near-plane and off-screen culling, the tile rectangle,
   and the SH colour for the view direction.
2. **Tile entries**: an exclusive scan of the per-splat tile counts sizes the entry buffer; each
   splat writes one 32-bit key (tile, then quantized depth) per tile it covers.
3. **Sort** the keys with their splat indices, once, globally.
4. **Tile ranges**: each tile's range of entries, from comparing neighbouring keys.
5. **Rasterize**: one 16×16 workgroup per tile blends its range front to back, in shared-memory
   batches, writing the colour, the final transmittance and the last contributor (kept for the
   backward pass), and an RGBA8 image.

Training's backward pass so far:

6. **Loss**: each pixel's loss against the camera's target image and its gradient with respect to
   the pixel's colour, summed.
7. **Backward rasterize**: each tile's entries walked back to front, from each pixel's last
   blended entry, giving each entry's gradients with respect to its splat's projection.
8. **Gather**: each splat's projection gradients, the sum of its entries'.
9. **Backward preprocess**: each splat's gradients with respect to its parameters (Appendix A's
   chain back through the conic, J·W, M = RS and the quaternion, the mean's projection and the SH
   colour), and the densification's statistics: its 2D-mean gradient norm and visible frames,
   accumulated.

Training will add Adam, and densification and pruning on a fixed capacity, compacted by scan.

## Conventions

- **Cameras**: COLMAP's, used directly in the kernels: (R, t) maps world to camera, +z looks
  forward, +y down, pixel centres at integer + 0.5. The camera centre is −Rᵀt.
- **Parameters**: position, log-scale, unit quaternion (w first), opacity logit and SH degree 3
  coefficients; the kernels apply the activations (exp, normalize, sigmoid).
- **Buffers**: std430 arrays of structs, so no kernel binds more than 8 storage buffers (WebGPU's
  default limit). Struct layouts are listed [below](#structs); the input files' `Splat` and
  `Camera` are in [data-format.md](data-format.md). No single buffer binds more
  than 128 MiB, lavapipe's `maxStorageBufferRange` and WebGPU's default
  `maxStorageBufferBindingSize` (cs5330's 46 images at 748² are about 103 MB); split a larger
  buffer, e.g. the images per batch. The rasterizer's `colours` (16 bytes a pixel) limits an
  image to 2²³ pixels; ZGAL can't check the product of two options, so the programs cap each side
  at 2896 (`--width`, `--height`), which also rejects some sizes within the budget, such as
  3840×2160 (8.3 Mpx).
- **Subgroups**: never assume a size. lavapipe's is 4, 8 or 16 depending on the CPU and MoltenVK's
  is 32 (the old sort broke on exactly this), so sorts and scans use ZGAL's `sort` and `scan`, and
  no kernel hard-codes 32 or 64.
- **Shared memory**: at most 16 KB per workgroup (WebGPU's default): rasterize batches 256 splats
  forward and 128 backward.
- **Capacity**: the splat count is bounded by a required `--max-splats`. ZGAL recreates a buffer,
  losing its contents, when its per-frame size changes, so only scratch buffers (tile entries)
  grow: the tile entries' capacity is a per-frame input the app grows after an overflow
  ([below](#render-graph)).

## Splat math

`lib/splat_math.zgal` holds what every kernel computes per splat. Its functions use no textures,
images or effects, so ZGAL also compiles them to inline C++ over GLM: the host and the CPU tests
compute exactly what the shaders do. `tests/cpp/splat_math_test.cpp` checks that C++ against the
closed-form oracles (`tests/oracle/closed_form.h`) and independent derivations.

- **Preprocess** (`projectSplat`): the camera-space mean `t = R p + t_cam`; Σ = M Mᵀ with
  M = R(q/|q|)·diag(exp(logScale)); Σ' = J W Σ Wᵀ Jᵀ + 0.3 I, with J evaluated at x/z and y/z
  clamped to the frustum guard (the reference code's 1.3 tan(fov/2), generalized to an off-centre
  principal point as [(−cx − 0.15 W)/fx, (1.15 W − cx)/fx]); the conic Σ'⁻¹; the radius
  ⌈3√λ⌉ with λ = mid + √max(mid² − det, 0.1), clamped to 2²⁰ before the cast to uint; the colour
  for the direction from the camera centre to the splat. A splat is culled (`tileCount` 0) when
  z ≤ `near` (a parameter; the reference code's 0.2), det Σ' isn't positive and finite (NaN from a
  zero quaternion or an overflowing exp, inf from a huge scale), or its tile rectangle is empty.
- **Tiles**: 16×16 pixels; a W×H image has ⌈W/16⌉×⌈H/16⌉. A splat's rectangle is
  [⌊(mean − r)/16⌋, ⌈(mean + r)/16⌉) on each axis, clamped to the grid (max exclusive): the tiles
  that meet the open square (mean − r, mean + r), so it covers every pixel centre within r of the
  mean in both axes. This departs from the reference code
  ([paper-conformance.md](paper-conformance.md), departure 7).
- **Sort keys** (departure 1 in [paper-conformance.md](paper-conformance.md)): one 32-bit key per
  tile entry, the tile index in the high `tileBits` = ⌈log₂ tileCount⌉ (1 to 31) bits and the depth
  in the low `depthBits` = 32 − tileBits. 748×748 has 2209 tiles: 12 tile bits, 20 depth bits. The
  depth, clamped to [`near`, `far`], is quantized by its float bits: a positive float's bits grow
  with its value, 2²³ steps per octave (exponent plus mantissa is log₂ to within 0.09 of an octave),
  so `bits(depth) − bits(near)`, shifted right by the least shift that fits `bits(far) − bits(near)`
  in depthBits, is close to log spaced. It uses integer operations only, so the GPU and the C++
  agree exactly; a log-based key wouldn't: a GPU's `log` needn't round as C++'s does, and an ulp can
  move a key across a step. With near 0.2 and far 1000, 20 depth bits give steps of at most 2⁻¹⁶ of
  the depth. The shift depends only on near, far and depthBits, so kernels keying many entries
  compute it once (`depthShift`, `depthKeyShifted`). The quantized depth is clamped to depthBits
  too, so not even a NaN depth reaches the tile bits. `depthKey` requires 0 < near < far and
  tile < 2^tileBits, unchecked. Keys tie for depths within one step and for all depths ≥ far, so
  ties need a rule for determinism: the render graph's ([below](#render-graph)).

## Render graph

`lib/render_graph.zgal`'s `SplatGraph` runs pipeline steps 1 to 5 for one camera per frame and,
with a window, draws the image on the surface.

1. `Preprocess`: `projectSplat` per splat into `projected`, and its tile count into `tileCounts`
   (0 when culled).
2. An `exclusive-scan` of `tileCounts` into `offsets` (each splat's first entry), and a `reduce`
   of their total into `counts`.
3. `PlanEntries` (one invocation): the entries this frame holds, min(total, capacity), the depth
   bits and shift (`depthShift`, so no entry recomputes it), and the indirect `DispatchCommand` for
   the entry kernels; `EmitEntries` then runs one invocation per entry, not per splat: entry e
   binary-searches `offsets` for its splat (the last whose offset is ≤ e), takes tile e − offset of
   the splat's rectangle (row by row), and writes its key and splat index.
4. A `sort` of the keys with the splat indices as values, over the entries held; a `clear` of
   `ranges`; `TileRanges` (the same indirect dispatch) writes each tile's [start, end) where
   neighbouring keys' tiles differ. Tiles without entries keep (0, 0). `ranges` is two `uint`s a
   tile (start at 2t, end at 2t + 1), not a `uvec2`, as different invocations write a tile's start
   and end ([web.md](web.md#portability)).
5. `Rasterize`: one 16×16 workgroup per tile (a 2D dispatch of the tile grid, at most 181² tiles
   for the programs' 2896² pixels, so no grid-stride loop and no `maxGroups`), one invocation per
   pixel. The workgroup walks the
   tile's range in batches of `rasterBatch` = 256 entries: each invocation loads one entry's
   splat (mean, conic and opacity, colour) into shared memory (10 KB of the 16 KB budget), a
   barrier, each pixel blends the batch front to back (Eq. 3 with Appendix C's thresholds, in the
   reference code's order: power = −½(a dx² + c dy²) − b dx dy with d = mean − pixel centre,
   skipped when positive; α = min(0.99, opacity·e^power), skipped below 1/255; the splat that
   would take T below 1e-4 isn't blended, and the pixel stops), then a barrier before the next
   batch overwrites it. The batch loop always covers the whole range (a barrier can't sit in a
   loop whose end depends on shared memory: [paper-conformance.md](paper-conformance.md#departures),
   5); a stopped pixel only helps load. Each pixel in the image writes:
   - `colours` (vec4 per pixel, row by row): C + T·bg, and T;
   - `lastEntries` (uint): the entry after the last one it blended, or the tile's start when none
     was, so the backward pass walks [start, last) back to front;
   - `rendered` (rgba8 storage image): C + T·bg clamped to [0, 1], alpha 1. Its values are
     the image's stored (sRGB-encoded) values, as the training images are, so no transfer
     function is applied.
   Pixels past the image's edge (in edge tiles) only help load batches, and every pixel inside
   writes all three, so nothing reads an unwritten value.
6. `present` (only with the `window` parameter): `Present` draws `rendered` on the surface as a
   quad, with nearest sampling, letterboxed by `presentScale` (`letterbox`, from the surface's
   size, so the image keeps its aspect ratio). The fragment shader decodes the colours
   (`srgbDecode`) so that an sRGB surface's encode gives back the captured values. **Known
   limitation:** this assumes the surface is sRGB. The web draws through an sRGB view and native
   Vulkan picks `B8G8R8A8_SRGB` when offered (MoltenVK does), but otherwise takes the surface's
   first format, and on a UNORM one the window shows the image darker, with no warning (ZGAL
   doesn't give a shader the surface's format). Captures are unaffected.

`counts` holds (entries, total, capacity, depth bits, depth shift) and is read back every frame.

The graph's inputs besides the scene: `background` (a per-frame vec3: a graph parameter can't be
a vec3), `window` and `capture` (whether to draw on the surface, and to read `rendered` back for
a PNG), `presentScale`, and `inspect` (read back every stage and the rasterizer's outputs, for the
tests). ZGAL has no graph composition, so the present pass lives in the graph behind `window`
rather than in the program.

- **Dispatch sizes**: a 1D dispatch has at most `maxGroups` workgroups (a required graph parameter;
  Rasterize's 2D tile grid is the exception, bounded by the image size: item 5):
  `groupLimit`, 65535, is what every device allows in X (Vulkan's guaranteed
  `maxComputeWorkGroupCount[0]`, lavapipe's, WebGPU's default `maxComputeWorkgroupsPerDimension`),
  and ZGAL can't query the device's own. Preprocess (64-wide), EmitEntries and TileRanges
  (256-wide) loop over their elements by grid stride, so 2²⁴ splats or 2²⁵ entries need no more
  groups. `PlanEntries` sizes the indirect dispatch with `groupsFor`; tests pass a smaller
  `maxGroups` (tile_test's `--max-groups`) so the loops go round.
- **Total**: the tile counts and their scan and total are `uint`, so the total must stay below
  2³² (e.g. 2M splats each covering all of a 748² view would wrap it). Past that the offsets
  aren't ascending and the frame's entries are wrong, but the kernels stay in bounds: an entry
  outside its splat's run gets key 0xFFFFFFFF and TileRanges writes no tile beyond the grid.
- **Capacity**: the entry buffers' size is a per-frame input (`capacity`), not a compile-time
  constant: a view's entry count isn't known before the preprocess. When a frame needs more, it
  keeps the first `capacity` entries (whole splats in index order, then part of one), so its
  image is incomplete, and the app, reading `counts`, grows the capacity with `grownCapacity`
  (the total plus 25%, so a slowly growing view doesn't regrow every frame) for the next frames.
  It never shrinks, and is at most 2²⁵ entries (128 MiB of keys, the binding limit above): a view
  needing more stays incomplete every frame, which `counts` (total > capacity) is the only sign
  of. ZGAL recreates the buffers when it changes, which is fine for scratch.
- **Determinism**: entries are written at fixed positions (offset + index in the splat's run), so
  before the sort they're in splat-index order, and ZGAL's `sort` is stable, so keys that tie
  stay in splat-index order. The blend order is the same on every run and every device.
- **GPU against C++** (`programs/tile_test`): the test recomputes stages 2 to 4 on the CPU from the
  GPU's own projections (means, radii, depths), so it compares the tile rectangles (`tileRect`'s
  floor and ceil), every key and splat position for position, and every range exactly. It checks a
  frame that overflowed its starting capacity of 16 (the first 16 entries in splat order, sorted,
  and their ranges), that the capacity grew, then a complete frame, some runs with `--max-groups` 1
  or 3. `tests/cpp/render_graph_test.cpp` checks `groupsFor`, `grownCapacity` and
  `grownCapacityUpTo` on the CPU. The preprocess itself is compared to float precision (mean 1e-4
  relative, depth 1e-5, radius ±1), since the GPU's `exp`, `sqrt` and division needn't round as
  C++'s do. Degenerate splats (a zero quaternion, logScale 100 and 30, appended by `scenes.py`;
  `--culled-tail`) must be culled on both; the `!(det > 0)` NaN guard holds on MoltenVK.
- **Rasterizer against C++** (`programs/render_test`): with a fixed `--capacity` (a frame needing
  more fails), every frame is complete and the same, so readbacks of different frames can be
  checked together. Every pixel's colour and T are compared within 1e-4 with a CPU blend in
  double of the GPU's projections over the GPU's tile ranges (checked by tile_test; render_test
  also checks that no range is inverted and that the non-empty ones partition the entries, which
  catches a lost start or end write on backends without a tile_test), and its last entry exactly;
  with `--oracle` (each S1 variant, every camera of the rotated ones), within 1e-3 with the
  closed-form oracle (`tests/oracle/closed_form.h`), recomputed from the splat's
  parameters and the camera's rotation about z, each splat blended only into its tile rectangle
  (which must equal the GPU's). A pixel with a threshold decision within rounding of its bound
  (α near 1/255, T near 1e-4, a nonzero exponent near 0) can go either way: it's borderline, and
  compared with the tolerance widened by what one flip changes (α·T, or T for the stop, times twice
  the largest colour) and its last entry only within the tile's range; at most 1% may be (0–3
  pixels are). The oracle takes the camera's principal point, so the S1 `off-centre` variant
  (48×32, (cx, cy) = (21, 12.5)) checks the non-square and principal-point conventions
  independently of the kernels' own projection (a width/height swap in Rasterize's row stride or
  the tile grid shows; one in covariance2d's frustum guard doesn't, as the clamp is never engaged
  and the oracle doesn't model it). Outputs must be finite and the same bit for bit
  over the next frames, and `test.sh` runs S2's cameras (cam_1 80×48, non-square) twice and
  compares the outputs' hashes.
  S3's cameras 0 and 24 (160×96) have tiles of up to about 480 entries (two batches) and pixels
  stopped by T < 1e-4. Observed differences are below 3e-7.
- **Test inputs**: ZGAL has no expression for a `buffer-file`'s element count and no path
  joining, so tile_test and render_test take the files, splat count and camera size as options;
  their `test.sh` reads them from the scene's `manifest.json` (`scenes.py`), and the checks verify
  them against the readbacks.

## Train graph

`lib/train_graph.zgal`'s `TrainGraph` runs the render graph's stages 1 to 5 for one camera per
frame (the same kernels and nodes, repeated: ZGAL has no graph composition, so there is no window
or capture), then pipeline steps 6 to 9. `splatGradients` (a `ProjectedGradient` per splat: the 2D
mean, the conic, the activated opacity and the colour) is overwritten every frame, so nothing
clears it.

6. `Loss` (256-wide, grid stride): per pixel, I = C + T·bg (the rasterizer's `colours`, unclamped)
   against the target, an RGBA8 pixel at `targetOffset` + the pixel's index in `targetFile`
   (an `images_<k>.bin`: [data-format.md](data-format.md)), which holds `targetPixels` pixels: a
   pixel past them reads black rather than past the buffer, and the app must check up front that
   `targetOffset` + W·H ≤ `targetPixels` (grad_test fails otherwise). With `loss` = `lossL2`, the
   loss is Σ(I − t)² over the channels (the gradient tests' smooth loss) and dL/dI = 2(I − t); with
   `lossL1`, |I − t| / 3WH and sign(I − t) / 3WH, so the image's loss is the mean (training's L1
   term). Per-pixel values go to `pixelLosses` and `pixelGrads`, and a `reduce` sums the loss into
   `losses[0]`.
7. `TileLasts` (64-wide, grid stride over the tiles) writes each tile's start, last and end into
   `backwardRanges` (3 uints a tile), the last being the largest of its pixels' `lastEntries`.
   `BackwardRasterize`: one 16×16 workgroup per tile, as Rasterize. Each pixel starts at its
   `lastEntries` with its final T and walks the tile's entries from the tile's last back to its
   start, in shared-memory batches of `backwardBatch` = 128 (half the forward's, so the batch and the reduction's scratch
   fit in 16 KB: 14,912 B). It undoes the blend as the reference code does: T ← T/(1 − α);
   dL/dc = αT·dL/dI; dL/dα = T(c − R)·dL/dI − T_final/(1 − α)·(bg·dL/dI), with R the colour
   behind the entry. dL/dα goes through α = min(0.99, opacity·G) to the opacity, mean and conic,
   with the clamp's exact derivative: 0 where it holds. The conic's b counts both off-diagonal
   entries (closed_form.h's convention). It skips exactly the entries the forward pass skipped
   (same computation). For each entry, the workgroup reduces its 256 pixels' 9 values in shared
   memory (256 → 16 partial sums → 1, in a fixed order) and one invocation per value writes it to
   `entryGradients` (9 floats an entry): no float atomics (D4(b)). The entries past the tile's
   last, which no pixel blended, get zeros.
8. `InvertEntries` writes each sorted entry's index at its place before the sort (its splat's
   offset plus its index in the splat's tile rectangle, row by row, recovered from the key's
   tile). `GatherGradients` (one invocation per splat) then sums its entries' gradients in tile
   order into `splatGradients`. Places no entry maps to (only after a wrapped total) keep the
   clear's 0xFFFFFFFF and are skipped, as are entries past the capacity.
9. `BackwardPreprocess` (64-wide, grid stride over the splats) runs `preprocessBackward`
   (`lib/splat_backward.zgal`) on each visible splat's `splatGradients`, recomputing its forward
   pass rather than storing it, into `parameterGradients`: a `Splat` of gradients (dL/dposition,
   the opacity logit's, the log-scale's, the stored quaternion's, and the 48 SH coefficients' up
   to `shDegree`, 0 past it; the padding isn't written), per frame and overwritten each frame. A
   culled splat gets zeros. The chain (the header of `preprocessBackward` has every formula):
   the conic's inverse to Σ′, Σ′ = J W Σ Wᵀ Jᵀ to Σ and to the camera-space mean through J (with
   x/z and y/z's clamp's exact derivative), Σ = M Mᵀ with M = R S (Eq. 11's dL/dM = 2 G M) to the
   log-scales and through R(q/|q|) to the stored quaternion, the mean's pinhole projection, and
   the SH colour (degrees 0 to 3, its clamp at 0, and its view direction's dependence on the
   position). `densifyStats` (a `DensifyStat` a splat, `#:frames shared`) accumulates over frames:
   each visible splat adds `densifyNorm` of its 2D-mean gradient (its norm in NDC units,
   |(W/2·∂L/∂x, H/2·∂L/∂y)|, the reference's) and 1 to its frame count. Nothing else writes it:
   the `#:once` clear `resetDensify` zeroes it in the first frame, and the densification reruns
   it (`run-once!`) after reading it. Two gaps, for the densification to close: the rerun is
   untested (grad_test only checks that every visible splat has the same count), and reading back
   then resetting in a later frame drops the frames in flight between the read and the reset
   from both intervals (1–2 frames of a 100-iteration interval; consuming and resetting in the
   same frame on the GPU would drop none). `tests/cpp/splat_backward_test.cpp` checks
   `preprocessBackward` on the CPU (the oracle and central differences of `projectSplat`).

`losses` holds (the loss, the probe, the loss's change) and is read back every frame, with
`counts`. With `inspect` (tests), so are `splatGradients`, `parameterGradients`, `densifyStats`,
the projections and `backwardRanges`, and the splats and cameras once.

- **Determinism**: every sum has a fixed order (the per-entry reduction, the per-splat gather, ZGAL's
  `reduce`), so a frame's gradients are the same bit for bit on every run and every frame. The
  price is in BackwardRasterize: every entry up to the tile's last costs the workgroup two
  barriers and a 9-value reduction, whether or not a given pixel blends it (the forward pass
  costs one barrier per 256 entries). `TileLasts` cuts the walk at the last entry any pixel
  blended, so a saturated tile's tail past every pixel's stop costs only its zero writes. Not yet
  measured against the forward pass on a real scene (T13's profiling).
- **Capacity**: the entry buffers are capped at 3·2²⁰ entries (`trainEntryCapacityMax`; a larger
  `capacity` counts as that), so `entryGradients` (36 B an entry) stays within one binding's 128 MiB
  (108 MiB). The rest of the render graph's capacity rules apply, but an app grows the capacity
  with `grownTrainCapacity` (`grownCapacityUpTo`'s rule at most `trainEntryCapacityMax`), and a
  frame needing more than the cap (`counts`' total past it) is truncated, the dropped entries'
  gradients lost, so a trainer must fail on it rather than step. **Open limit**: a real scene at
  full resolution can need 10M+ entries. Splitting `entryGradients` can't reach 2²⁵ within
  WebGPU's 8 storage buffers (BackwardRasterize binds 7); processing the tiles in chunks past the
  cap is left for when the trainer's profiling shows real scenes need it.
- **Finite differences** (`finiteDifferences`, for grad_test): `Perturb` adds `perturbDelta` to
  one field (`perturbField`: opacity, a colour channel, the mean's x or y) of one splat's
  projection after the preprocess, so the tiles stay the same; or, with `perturbField` =
  `perturbParameter` plus a float offset in `Splat`, to one of the splat's parameters, projecting
  it again (`projectSplatPerturbed`, which also takes an SH coefficient) and keeping the new
  mean, conic, opacity and colour but the old depth and tiles, so the sort and the entries don't
  change (nor would the analytic gradients see them change). A re-projection that culls the splat
  keeps the old projection. A probe k runs its +h frame
  (`probe` 2k + 1), then its −h frame (2k + 2); probe 0 is unperturbed. `LossDelta` keeps the +h
  frame's per-pixel losses and `lastEntries` (`plusLosses`, `plusLasts`, `#:frames shared`), and
  the −h frame writes each pixel's plus − minus, reduced into `losses[2]`: the difference is
  summed pixel by pixel, not taken between two whole losses. Every frame writes its probe into
  `losses[1]`, so the app matches a late readback with its probe. A pixel where a decision flips
  between the sides counts nothing, since the loss jumps there and the analytic gradient is the
  derivative between the jumps: the perturbed splat's blend (α against 1/255, the exponent's
  sign), or the stop at T < 1e-4 moving to another entry (any splat's: the pixel's
  `lastEntries` differ). The +h frame keeps the perturbed projection too (`plusProjected`, one
  element), so the −h frame compares the splat's blend decision on both sides' projections.
- **Tests** (`programs/grad_test`): with `--oracle` (S1's `gradient` variant: one splat covering
  the image, a constant target), the GPU's projection gradients and loss must be within 1e-3
  relative of `projected_gradients` (L2 and L1) on the oracle's projection. With
  `--finite-differences`, central differences on the GPU must agree within the plan's rule
  ([building-and-testing.md](building-and-testing.md#gradient-checks)) where the oracle can't
  reach: S2's cameras (cam_1 is 80×48, non-square), S1's `one-splat-clamped` (the clamp's
  derivative), S1's `stack` (the stop at T < 1e-4) and S3's cameras 0 and 24 (tiles past one
  backward batch); with `--parameters`, the parameters' gradients the same way on S2's cameras and
  S3's camera 0. The S2 L1 run with `--max-groups 3` (so the grid-stride loops go round) runs
  twice, and the hashes are compared. Every run checks the gradients are finite; the others also
  check the next frames' gradients are the same bit for bit.

## splat_render

`programs/splat_render` runs `SplatGraph` for one camera: in a window by default (escape quits;
`--size` sets the window, and the image is letterboxed in it), or headless with `--frames N`,
`--capture FILE` writing the last frame's `rendered` as a PNG. ZGAL can't read the manifest or
join paths, so the scene comes as options, which `tools/scene_args.py <scene dir> <camera>`
prints (the camera an index or a name):

```
build/splat_render/splat_render $(tools/scene_args.py <scene dir> <camera>) --frames 8 --capture out.png
```

`--background` is a grey level. The tile entries' capacity starts at `--initial-entries` (2²⁰)
and grows after a frame that overflowed it (`grownCapacity`, with a message): that frame and
those already in flight miss splats, so a capture needs enough frames after the growth (the
readback arrives a few frames later) or a large enough start. Its `test.sh` renders the
committed S3 scene (`programs/splat_render/golden/s3`, `make_synthetic.py s3` with seed 0,
committed so a change in numpy's random streams can't move it) from cameras 0 and 24, one starting
at 16 entries so it must grow, and compares each capture with the reviewed golden
(`golden/s3-camera-<k>.png`) at a PSNR of at least 45 dB (`psnr.py`).

### Structs

std430; `Splat` and `Camera` are the input files' ([data-format.md](data-format.md)), repeated
here with their offsets.

```
(struct Splat              ; 240 bytes
  (position vec3)          ; 0: world space
  (opacity float)          ; 12: logit
  (logScale vec3)          ; 16: then 4 bytes of padding
  (rotation vec4)          ; 32: quaternion (w, x, y, z) in (.x, .y, .z, .w), not normalized
  (sh (array float 48)))   ; 48: coefficient k (0-15) of channel c (RGB) at sh[3k + c]

(struct Camera             ; 112 bytes
  (worldToCamera mat4)     ; 0: column-major; [R t; 0 1], X_cam = R X + t
  (intrinsics vec4)        ; 64: (fx, fy, cx, cy) in pixels
  (center vec3)            ; 80: the camera centre in world space, −Rᵀt
  (pixelOffset uint)       ; 92: the image's first pixel in its batch
  (size uvec2)             ; 96: (width, height)
  (imageBatch uint)        ; 104: which images_<k>.bin; 0xFFFFFFFF without an image
  (heldOut uint))          ; 108: 1 for a test camera (the manifest's split), else 0

(struct Projected          ; 64 bytes: one splat as one camera sees it (the preprocess's output)
  (mean vec2)              ; 0: the 2D mean, in pixels
  (depth float)            ; 8: camera-space z
  (radius uint)            ; 12: the 3σ radius, in pixels
  (conic vec3)             ; 16: Σ'⁻¹ as (a, b, c) for [[a, b], [b, c]]
  (opacity float)          ; 28: sigmoid(Splat.opacity)
  (colour vec3)            ; 32: the SH colour for this camera's view direction
  (tileCount uint)         ; 44: (maxX − minX)(maxY − minY); 0 when culled
  (tiles uvec4))           ; 48: the tile rectangle (minX, minY, maxX, maxY), max exclusive

(struct DensifyStat        ; 8 bytes: a splat's densification statistics (train graph)
  (gradientNorm float)     ; 0: the sum of densifyNorm(dL/dmean) over the frames that saw it
  (visible uint))          ; 4: those frames' number
```
