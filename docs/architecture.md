# Architecture

## Layout

- `programs/<name>/`: one ZGAL program each (`<name>.zgal` with a `main`, plus its own `.cpp` and
  `.ts` files, `assets/` and optional `test.sh` / `test-web.sh`). The Makefile finds them; adding
  one needs no Makefile change ([building-and-testing.md](building-and-testing.md)).
  - `splat_render`: renders a camera of a scene in a window, or headless with a PNG capture
    ([below](#splat_render)).
  - `splat_train`: trains a scene's splats against its images, headless, scores the held-out
    views and writes an INRIA PLY ([below](#splat_train)).
  - `tile_test` and `render_test`: the render graph's GPU tests against the CPU and the
    closed-form oracles ([below](#render-graph)). render_test's checks are in `render_check.h`,
    shared by the native program and `web-check/render_test_check.cpp`, which checks a web run's
    readbacks ([web.md](web.md#tests)).
  - `grad_test`: the training graph's gradients against the closed-form oracle and GPU finite
    differences ([below](#train-graph)); its checks are in `grad_check.h`, shared with
    `web-check/grad_test_check.cpp` as render_test's are.
  - `densify_test`: one densification frame (clone, split, prune, compaction, the counts, the
    opacity reset) against the CPU, every check in the app ([below](#train-graph)).
  - `ssim_test`: the SSIM kernels against a float64 numpy reference (an image against itself,
    constant images, a random pair) and GPU finite differences, every check in the app
    ([below](#train-graph)).
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
    ([below](#train-graph)); it imports `render_graph.zgal` and `ssim.zgal`.
  - `ssim.zgal`: the D-SSIM term's forward and backward kernels (a separable 11×11 Gaussian
    window), which add to the Loss kernel's per-pixel losses and gradients
    ([below](#train-graph)).
  - `splat_stream.zgal`: the stream format's header, its check and its decode
    ([data-format.md](data-format.md#stream-files)); it imports `splat_math.zgal`.
  - `view_camera.zgal`: the viewer's camera controls, a pose relative to a start camera turned
    by mouse deltas, moved by keys and fitted to the window's size ([below](#view-camera)); it
    imports `splat_math.zgal`.
  - `native/`: header-only C++ that the data tools share with the programs, on top of the C++ ZGAL
    generates for those modules: the stream encoder (`stream_encoder.h`), the default camera it
    writes into the header (`default_camera.h`, [data-format.md](data-format.md#stream-files)) and
    the INRIA PLY reader (`ply_reader.h`, [data-format.md](data-format.md#inria-ply)) and the
    splats.bin reader (`splats_bin_reader.h`). `c_api.cpp` wraps them in a C ABI for the tools'
    ctypes binding (`tools/splatdata/native.py`), built into `build/native/libsplat_native`
    ([building-and-testing.md](building-and-testing.md#the-native-library)).
    ZGAL puts the generated code in the namespace of the importing `.zgal` file, so an includer
    aliases it first: `#include "<x>.zgal.h"`, then `namespace splat_native { namespace gen = ::<x>; }`,
    then the header. Entry points take plain structs and spans and return status codes (no
    exceptions), so a C ABI can wrap them. Their functions are `inline` in `splat_native`, so a
    linked binary must alias `gen` to one namespace in every translation unit (otherwise it's an
    ODR violation).
- `tools/`: offline Python (numpy, scipy, Pillow): the converter from COLMAP text and INRIA PLY files to the
  programs' inputs, the synthetic scene generator (which also renders a scene's ground truth with
  splat_render), and `scene_args.py`, which prints splat_render's arguments for a scene and a
  camera, or splat_train's (`--train`) for a scene with images.
- `tests/`: C++ tests (`tests/cpp/`), closed-form oracles (`tests/oracle/`). The Python tests are in
  `tools/tests/`.
- `colmap/`: the one real scene: `colmap_exports3`, a COLMAP text export (one SIMPLE_RADIAL
  camera, 46 images, sparse points; CRLF line endings), and its images, `cs5330_dense_resized`
  (748×748 JPEGs, named as in `images.txt`), committed directly (6.2 MB, no LFS).
- `external/vulkan_test`: the wrapper and ZGAL.

## Pipeline

Steps 1 to 5 are `lib/render_graph.zgal` ([below](#render-graph)), and steps 6 to 9 are
`lib/train_graph.zgal` ([below](#train-graph)), with step 9's math in `lib/splat_backward.zgal`.

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

Training's backward pass:

6. **Loss**: each pixel's loss against the camera's target image and its gradient with respect to
   the pixel's colour, summed.
7. **Backward rasterize**: each tile's entries walked back to front, from each pixel's last
   blended entry, giving each entry's gradients with respect to its splat's projection.
8. **Gather**: each splat's projection gradients, the sum of its entries'.
9. **Backward preprocess**: each splat's gradients with respect to its parameters (Appendix A's
   chain back through the conic, J·W, M = RS and the quaternion, the mean's projection and the SH
   colour), and the densification's statistics: its 2D-mean gradient norm and visible frames,
   accumulated.

Training adds Adam, and densification and pruning on a fixed capacity, compacted by scan
([Train graph](#train-graph)).

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
- **Splat count**: the count of splats in use lives on the GPU, in `splatCounts` (`#:frames
  shared`; `splatActive` is its first `uint`), because training's densification changes it on the
  GPU. `InitSplatCounts` (`#:once`) writes the scene's `splatCount` there and the indirect
  `DispatchCommand` in `splatCommands` (`groupsFor` with `maxGroups`) in the first frame; the
  per-splat kernels read the count from `splatCounts` and are dispatched indirectly through
  `splatCommands`, and the scan and reduce over `tileCounts` take their count from it on the GPU.
  `SplatGraph`'s splats never change, so it sizes its buffers with `splatCount`; `TrainGraph`'s
  per-splat buffers hold `maxSplats` (a parameter, at least `splatCount`): ZGAL recreates a buffer
  whose size changes, losing its contents, so the capacity is fixed and only the count moves.
  `TrainGraph` copies the scene's splats (`splatSource`, the file) into the first `splatCount` of
  its `splats` buffer once (`loadSplats`), and the splats past the count are never read.
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

The camera (`cameraIndex`), its size (`width`, `height`) and `shDegree` are per-frame inputs, so a
trainer renders another camera every frame. The per-pixel and per-tile buffers are sized once, to
`maxWidth` × `maxHeight` (at least every camera's size), and this frame's grid, tile and pixel
counts are computed in the push constants, `#:invocations` and `reduce` counts that use them
(a graph `constant` is computed once, when the graph is built, so it can't follow a per-frame
input).

6. `Loss` (256-wide, grid stride): per pixel, I = C + T·bg (the rasterizer's `colours`, unclamped)
   against the target, an RGBA8 pixel at `targetOffset` + the pixel's index in `targetFile`
   (an `images_<k>.bin`: [data-format.md](data-format.md)), which holds `targetPixels` pixels: a
   pixel past them reads black rather than past the buffer, and the app must check up front that
   `targetOffset` + W·H ≤ `targetPixels` (grad_test fails otherwise). With `loss` (per frame) = `lossL2`, the
   loss is Σ(I − t)² over the channels (the gradient tests' smooth loss) and dL/dI = 2(I − t); with
   `lossL1`, |I − t| / 3WH and sign(I − t) / 3WH, so the image's loss is the mean (training's L1
   term); with `lossEval` (a held-out view's score), (clamp(I, 0, 1) − t)² / 3WH and no gradient,
   so the image's loss is its MSE on the clamped render (PSNR's). Per-pixel values go to
   `pixelLosses` and `pixelGrads`, and a `reduce` sums the loss into `losses[0]`.
   With the `dssim` parameter, the `lossL1` frames' loss is the paper's (1 − λ)·L1 + λ·D-SSIM
   (λ the per-frame `dssimWeight`, splat_train's `--lambda-dssim`, 0.2): Loss scales its L1 term
   by 1 − λ (a push constant), and `lib/ssim.zgal`'s four kernels (256-wide, grid stride) add
   λ(1 − S) / 3WH a pixel and channel to `pixelLosses` and its gradient to `pixelGrads`, before
   the reduce. S is the reference's SSIM (per channel, the 11×11 Gaussian window with σ 1.5,
   zero-padded at the edges, C1 = 0.01², C2 = 0.03², on the unclamped render). The window is
   separable, so `SsimRows` blurs x, y, x², y² and xy along the rows and `SsimColumns` along the
   columns, giving S and the loss's partial derivatives with respect to each pixel's blurred
   mean, mean square and product; `SsimGradRows` and `SsimGradColumns` blur those partials back
   (the window is symmetric) into dL/dx = G∗∂μ + 2x·G∗∂(x²) + y·G∗∂(xy). Other frames (`lossL2`,
   `lossEval`) pass λ = 0 to them, which makes them return at once. Their scratch (7 vec4s a
   pixel at `maxWidth` × `maxHeight`, 112 bytes) is `#:frames shared` (training's frames are
   serial: each steps the splats the next renders), and without `dssim` none of it is allocated.
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
   last, which no pixel blended, get zeros. Steps 7's BackwardRasterize and 8's GatherGradients
   run in chunk passes of `chunkEntries` sorted entries each (see Capacity below); pass c writes
   only the entries in [c·chunkEntries, (c+1)·chunkEntries), at their offset in the chunk, and
   walks a tile only from its last down to the chunk's first entry (the entries after the chunk
   still undo the blend, so each pixel's T and R are what one pass would give).
8. `InvertEntries` writes each sorted entry's index at its place before the sort (its splat's
   offset plus its index in the splat's tile rectangle, row by row, recovered from the key's
   tile). `GatherGradients` (one invocation per splat) then sums its entries' gradients in tile
   order into `splatGradients`. Places no entry maps to (only after a wrapped total) keep the
   clear's 0xFFFFFFFF and are skipped, as are entries past the capacity. In a chunk pass, a
   splat's entries in the chunk are a contiguous run of its places (they share the depth, so they
   sort by tile, the order of its places), found by binary search; the sum continues from the
   previous pass's `splatGradients` (pass 0 starts at 0), so the additions and their order are
   one pass's and the result is the same bits.
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
   |(W/2·∂L/∂x, H/2·∂L/∂y)|, the reference's) and 1 to its frame count, and keeps its largest
   `Projected` radius (`maxRadius`, the reference's `max_radii2D`). The `#:once` clear
   `resetDensify` zeroes it in the first frame, and each densification (step 12) clears it in its
   own frame, after reading it on the GPU, so no frame's statistics are lost or counted twice
   (`resetDensify` is never rerun). `tests/cpp/splat_backward_test.cpp` checks
   `preprocessBackward` on the CPU (the oracle and central differences of `projectSplat`).
10. `Adam` (64-wide, grid stride over the splats; only with the `train` parameter) steps every
    splat's parameters with `parameterGradients` (`adamStep`, four floats at a time: β₁ = 0.9,
    β₂ = 0.999, ε = 1e-15, the reference's), keeping the moments in `adamM` and `adamV` (a `Splat`
    each, `#:frames shared`, zeroed by `#:once` clears in the first frame). The app pushes the
    bias corrections (1 − β₁ᵗ, 1 − β₂ᵗ, computed by the app, so every backend steps the
    same) and the learning rates per group: the position's, the SH DC's, the rest of the SH's,
    the opacity logit's, the log-scales' and the quaternion's. The padding isn't stepped. A frame
    whose entries overflowed (`counts`' total past its capacity, so gradients were lost) steps
    nothing; the app grows the capacity and the next frame steps.
11. `Record` (one invocation, with `train`) writes `losses[0]` into `history[historySlot]` (−1
    for a frame Adam skipped; `historySize` floats, `#:frames shared`, zeroed once), so the app
    reads every iteration's loss once at the end: a readback event sees only the newest frame's
    copy, and a trainer runs frames back to back.
12. **Densification and pruning** (with `train`; Sec. 5.2, the reference's `densify_and_prune`):
    nodes off (`#:enabled #f`) until the app enables them all in a densification's frame, after
    the backward pass and instead of Adam (the reference's step skips the parameters a
    densification replaced). The splats in use are the first `splatCounts[splatActive]` of
    `splats` ([render graph](#render-graph)'s splat count), rewritten in place:
    - `DensifyPrepare` (one invocation) writes the scan's count, 2n + 2 for n splats, into
      `splatCounts[splatSlots]`, and zeroes `slots[n]` and `slots[2n + 1]`.
    - `DensifyMark` (indirect, per splat) takes each splat's action (`densifyAction`) into
      `actions` and its flags into `slots`: `slots[i]` = 1 unless pruned, `slots[n + 1 + i]` = 1
      for a clone or a split. The actions: **prune** when its opacity is below 0.005, or, with
      `screenPrune` (the app's: after the first opacity reset), its largest scale past
      `pruneScale` (0.1 × extent) or its largest radius past `maxScreenSize` px (the reference's
      20, which its code never applies; 0 for none: [departure
      12](paper-conformance.md#departures)); otherwise, when its mean
      gradient (`densifyGradient`: the norm sum over its frames, 0 for none) is at least
      `densifyThreshold`, **clone** when its largest scale is at most `cloneScale`
      (`percent_dense` × extent), else **split**; else **keep**. A split's halves are checked
      against `pruneScale` at their own scale (1.6 times smaller) and not by radius (theirs isn't
      known); a pruned splat neither clones nor splits (its clone or halves would be pruned too).
    - An in-place `exclusive-scan` of the 2n + 2 flags (`#:count` from `splatCounts`) gives each
      kept splat its new index (`slots[i]`, in order), the kept number K (`slots[n]`), each
      addition its index K + the additions before it (`slots[n + 1 + i]`: after the survivors, in
      their parents' order) and the total (`slots[2n + 1]`).
    - `copy` the splats into `parameterGradients` (dead after the backward pass, so the
      densification's scratch costs no memory), then `DensifyScatter` (indirect) writes each kept
      splat at its index, a clone's copy at its addition's, and a split's halves at both: each a
      sample of the splat's Gaussian, R(scale ⊙ n) + position with n three standard normals
      (`standardNormals`: Box-Muller on `pcgHash`es of `splitKey`, from the app's `densifySeed`,
      the splat's index and the half), and the log-scale of the scale / 1.6. Then the same for
      `adamM` and `adamV` (`DensifyScatterMoments`, each through `parameterGradients`): a kept or
      cloned splat keeps its moments, a clone's copy and both halves start at 0.
    - An addition at or past the capacity (`maxSplats`) is dropped: a clone isn't made and a split
      isn't done (the splat is kept as it was, with its moments). `DensifyFinish` (one invocation)
      sets the count to min(total, capacity), adds the dropped ones to `splatOverflow`, counts the
      densification (`splatDensifies`) and writes its pruned and added numbers (`splatPruned`,
      `splatAdded`) and the per-splat `DispatchCommand` for the new count.
    - `densifyClear` zeroes `densifyStats`.
    `ResetOpacity` (`resetOpacity`, enabled separately in a reset's frame, after any
    densification) sets every splat's opacity logit to at most logit(0.01) (`resetOpacityLogit`)
    and zeroes its Adam moments, as the reference's `reset_opacity` does. `splatCounts` is read
    back every frame (`readSplatCounts`). `tests/cpp/densify_test.cpp` checks the actions and the
    split's samples on the CPU, and `programs/densify_test` the kernels on the GPU.

`losses` holds (the loss, the probe, the loss's change) and is read back every frame, with
`counts` and `splatCounts`. With `inspect` (tests), so are `splatGradients`, `parameterGradients`, `densifyStats`,
the projections and `backwardRanges`, and the splats and cameras once; with `inspect` and `train`,
`adamM`, `adamV`, `actions` and `slots` (densify_test's). With `train`, the cameras
are read too, and `history` and the trained splats when the app enables `readHistory` and
`readTrained`.

- **Determinism**: every sum has a fixed order (the per-entry reduction, the per-splat gather, ZGAL's
  `reduce`), so a frame's gradients are the same bit for bit on every run and every frame. The
  price is in BackwardRasterize: every entry up to the tile's last costs the workgroup two
  barriers and a 9-value reduction, whether or not a given pixel blends it (the forward pass
  costs one barrier per 256 entries). `TileLasts` cuts the walk at the last entry any pixel
  blended, so a saturated tile's tail past every pixel's stop costs only its zero writes. On a
  real scene it costs 4× the forward rasterize ([splat_train](#splat_train)'s numbers).
- **Capacity**: two required parameters. `maxEntries` is the most entries a frame may hold (at
  most 2²⁵, the 128 MiB binding of the 4-byte keys, sort and entry buffers; a larger `capacity`
  counts as it), and `chunkEntries` the entries one backward pass holds (at most 3·2²⁰,
  `trainChunkEntriesMax`, so `entryGradients`, 36 B an entry, fits one binding: 108 MiB).
  `entryGradients` holds one chunk; the graph unrolls ⌈maxEntries / chunkEntries⌉ chunk passes (a
  graph `for` of BackwardRasterize and GatherGradients nodes, each indirect), and `PlanChunks` (one
  invocation, after the sort) writes their dispatch commands from the frame's `counts`: the
  first ⌈entries / chunkEntries⌉ passes (at least one) get the full grid and the splats' groups,
  the rest zero groups, so a frame runs only the passes it needs, with no host round trip or
  extra submission. A frame that fits one chunk gets the same bits as one pass, at the cost of
  `PlanChunks`, the unused passes' empty dispatches and the gather's binary search. Every pass
  the graph holds is recorded every frame, an unused one as two zero-group dispatches and their
  barriers, so the passes are at most 64 (`trainChunkPassesMax`): enough for chunks down to 2¹⁹
  at 2²⁵ entries, and for the tests' dozens of small ones. TrainGraph can't fail, so past 64 it
  holds only the first 64 passes, and an app refuses such parameters up front with
  `trainChunkPasses` (splat_train's and grad_test's first frame). Splitting
  `entryGradients` into bindings instead couldn't pass 2²⁵ within WebGPU's 8 storage buffers
  (BackwardRasterize binds 7) and would cost the memory of a frame's whole entries at 36 B each.
  The rest of the render graph's capacity rules apply; an app grows the capacity with
  `grownCapacityUpTo` at most `maxEntries`, and a frame needing more than `maxEntries` (`counts`'
  total past it) is truncated, the dropped entries' gradients lost, so a trainer must fail on it
  rather than step. Past 2²⁵ would need the sort and the keys chunked too.
  Memory: each entry of the capacity costs 20 B per frame in flight (the keys, the splat indices
  and the positions, 4 B each, and the sort's scratch copies of the first two, 8 B), and
  `entryGradients` 36 B per entry of the smaller of the capacity and a chunk. At splat_train's
  default `--max-entries`, 2²⁵, a frame needing 25M entries (75% of it) grows the capacity to
  2²⁵: 640 MiB per frame in flight, plus 108 MiB per frame of `entryGradients`. `--max-entries`
  (and `--chunk-entries` for the gradients) is how an app caps that memory.
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
  With `--l1 --dssim 0.2`, the finite differences of training's 0.8·L1 + 0.2·D-SSIM on S2's
  cameras, probing the colour alone (the D-SSIM window spreads a flipped pixel's jump to its
  neighbours; [building-and-testing.md](building-and-testing.md#gradient-checks) says why).
  densify_test builds `TrainGraph` without `dssim`.
- **SSIM tests** (`programs/ssim_test`): the four SSIM kernels alone (λ = 1) on pairs from its
  `make_pair.py`, against the float64 numpy SSIM: an image against itself (32×32, within 1e-5;
  SSIM 1 and every gradient 0), constant images (32×32, within 2e-4; every pixel whose window lies inside the image has the
  closed-form luminance term (2ab + C1) / (a² + b² + C1)) and a random 40×24 pair (within 1e-5; non-square, so a
  kernel swapping the width and height fails) whose gradients must
  agree with GPU central differences, Richardson-extrapolated from h and 2h (32 probes, the
  corners first, then `pcgHash`-seeded pixels and channels, so every platform probes the same).

## View camera

`lib/view_camera.zgal` holds the viewer's camera controls as pure functions, so the native and
web apps (C++ and TypeScript) and the shaders compute the same camera. The viewer's state is a
`ViewPose` relative to a start `Camera` (from a cameras file, or the stream header's default
camera): the offset of the camera centre from the start's (world space), a yaw and a pitch, all
zero for the untouched pose (`viewPose`). Up is the start camera's −y (COLMAP: image y down).

- `viewLook(pose, delta, radiansPerPixel)` turns by a cursor delta in pixels: +x turns right
  (yaw, about the start's up, in world space), +y looks down (pitch, about the turned camera's
  x axis). The yaw wraps to [−π, π) (to float rounding: a few ulps outside it can occur) and the pitch is clamped to ±`viewPitchLimit` (89°). Since
  the view depends on (yaw, pitch) only, the camera never rolls: its x axis stays perpendicular
  to the start's up after any looks.
- `viewMove(pose, start, keys, span)` moves `span` world units along `keys` = (right, up,
  forward), e.g. (D − A, E − Q, W − S): right and forward are the current camera's x and z
  (forward follows the pitch), up is the start's, so Q and E stay level. The direction is
  normalized; keys that cancel don't move.
- `viewSpeed(baseSpeed, steps, stepFactor)` is baseSpeed · stepFactor^steps, computed from the
  scroll step count, so scrolling back gives the same speed exactly. The caller picks
  baseSpeed (world units per second) and stepFactor.
- `fitCamera(pose, start, size)` is the camera to render at `size`: the start's rotation
  followed by the pose's turn, centred at the start's centre plus the offset, and one scale
  s = H′/H for the intrinsics: fx′ = s·fx, fy′ = s·fy, cx′ = W′/2 + s·(cx − W/2) and
  cy′ = H′/2 + s·(cy − H/2) (computed as s·cx + (W′ − s·W)/2, exact for s = 1). That keeps the
  vertical field of view and square pixels; a wider window sees more horizontally. The
  translation is built from the start's, R_turnᵀ(t₀ − R₀·offset), not from the centre, so the
  untouched pose at the start's size gives the start camera exactly (its other fields, such as
  `imageBatch`, are kept). Both heights must be non-zero (skip rendering a minimized window).

`tests/cpp/view_camera_test.cpp` checks that identity bit for bit (the default camera
and random ones), the fit's scaling (square pixels, points at W′/2 + s·(u − W/2), the way back),
the directions of looks and moves, their round trips, the yaw's wrap and the pitch's clamp, that
the x axis stays level over thousands of random looks, and the speed's exact way back.

## splat_train

`programs/splat_train` runs `TrainGraph` (with `train`) headless, one Adam iteration a frame, for
`--iterations` iterations, then scores the held-out views and writes the trained splats as an
INRIA PLY (to `$SPLAT_TRAIN_PLY`, `splat_train.ply` by default, through the `exportPly` extern:
the `.cpp` natively, Node's `fs` on the web). `tools/scene_args.py --train <scene dir>` prints the
scene's options, and refuses a scene older than format version 2 or whose camera buffer's split
disagrees with its manifest (splat_train reads the split from the buffer, so a version-1 scene
would train on every camera and silently report no held-out views); `--frames` must cover the
iterations, the held-out views and a few frames of loading and readback latency, or the program
fails.

```
build/splat_train/splat_train $(tools/scene_args.py --train <scene dir>) --max-splats 100000 \
    --max-screen-size 20 --iterations 2000 --frames 2100
```

`--max-screen-size` is required too: the screen-space prune's radius in pixels after the first
reset, 20 as the reference's code means to or 0 (none) as it does, since it clears the radii
before reading them ([departure 12](paper-conformance.md#departures)). A run without it fails
while loading.

`--max-splats` is required (it has no default, since it trades memory for how far the scene may
grow): the capacity of every per-splat buffer, at least `--splat-count` and at most 559240 (the
128 MiB binding budget over a 240-byte `Splat`). The scene's splats are copied into the first
`--splat-count` of the splat buffer, and only the splats in use are trained and written.

Its phases, in one graph:

1. **Loading**: the first frames read the cameras back (Adam and Record off; they render camera 0
   at 16×16, or smaller within `--max-width` × `--max-height`). The app checks them
   against the options (the camera count, `--max-width` × `--max-height`, a single image batch,
   the targets' size, `--max-splats` ≥ `--splat-count`) and splits them by `heldOut` into
   training and test views (a camera without an image is neither).
2. **Training**: iteration t (1-based) renders a training camera popped from a stack of them,
   refilled when empty (the reference's order; the pick is `pcgHash(pcgHash(seed) + t)` modulo the
   stack's size, so every platform trains the same order), takes the gradients of the loss
   (1 − λ)·L1 + λ·D-SSIM (`--lambda-dssim` λ, 0.2; 0 trains on L1 alone and skips the SSIM
   kernels), and
   steps Adam (bias corrections 1 − 0.9ᵗ and 1 − 0.999ᵗ, computed by the app) at SH degree
   min(`--sh-degree`, t / `--sh-interval`). The position's learning rate decays log-linearly from
   `--position-lr` to `--position-lr-final` over `--position-lr-steps`, times `--extent`; the
   others are constant ([paper-conformance.md](paper-conformance.md#values-from-the-reference-code)
   has the defaults). Record writes the loss into `history[t − 1]`. Densification follows the
   reference's schedule: iteration t densifies when `--densify-from` < t < `--densify-until` and
   t is a multiple of `--densify-interval` (500, 15000, 100), with the threshold
   `--densify-threshold` (0.0002), the clone scale `--percent-dense` (0.01) × `--extent`, the
   prune scale 0.1 × `--extent` and radius `--max-screen-size`, the size prunes once
   t > `--reset-interval` and a new seed each
   time (`pcgHash(pcgHash(seed + 1) + t)`). A densifying frame doesn't step Adam (the reference's
   step finds no gradients on the new tensors, so it does nothing), and Adam's bias corrections
   count the steps taken (t less the densifications). When t < `--densify-until` is a multiple of
   `--reset-interval` (3000), or t = `--densify-from` with a white background, the frame also
   resets the opacity. The app prints each densification's counts as `readSplatCounts` brings
   them (`splat_train: densification K: N splats (P pruned, A added; O dropped for --max-splats
   so far)`). The last iteration neither densifies nor resets: the reference scores and saves an
   iteration before its densification and reset ([departure 17](paper-conformance.md#departures)).
   A run ending soon after a reset still scores nearly reset opacities, as the reference would.
3. **Evaluating**: each test view renders once with `lossEval` (the MSE of the clamped render, at
   the final SH degree; backward and Adam off) into `history[iterations + k]`.
4. **Reading**: the history and the trained splats are read back (a readback event sees only the
   newest copy, so the losses accumulate on the GPU and are read once). The app prints each test
   view's PSNR (10 log₁₀(1/MSE)) and their mean, the first and last epochs' mean training loss (an
   epoch is as many iterations as there are training views) and their ratio, the iterations
   skipped, and the splats in use (`splatCounts[splatActive]`, once the counts read back show
   every densification), the densifications and the additions dropped for `--max-splats`; then
   writes the splats in use as the PLY, and fails if a loss or a splat isn't finite.

The tile entries' capacity starts at `--initial-entries` and grows (`grownCapacityUpTo`, at most
`--max-entries`, default and most 2²⁵) after a
frame that needed more, from the counts read back after iteration 1's frame (loading's frames
train nothing, and this way iteration 1 runs at `--initial-entries` whether a platform delivers a
frame's reads before or after its frame event). On the web a tick's reads apply before its frame
event, so iteration 1's own counts are dropped: with `--iterations 1`, an overflowing or over-cap
iteration 1 neither grows the capacity nor fails the run, it is only reported as skipped (a later
pick of the same view is checked, so any longer run is covered). Adam skips a frame whose entries overflowed (its
gradients are incomplete), so that iteration is lost (history −1) and counted in the report. A
frame needing more than `--max-entries` entries, which no growth can hold, fails the
program, and so does an overflowing test view: each renders once, so a held-out camera needing
more entries than any training view grew the capacity to fails the run after its training, with
the `--initial-entries` to rerun with (the most a frame was seen to need, grown). Retrying the
view once the growth arrives would need the evaluation to wait on the counts' readback per view; a
follow-up if real scenes hit it. The splats train in place in the splat buffer (the file isn't
written).

Its `test.sh` gives the committed S3 scene a ground truth with splat_render
(`make_synthetic.py render-gt --sh-degree 0`, the same build: native or web) and trains twice on
it at 128×128 with the reference's learning rates and densification schedule (`--max-screen-size
20`), the SH degree rising every 500, but densifying until 1500 only (`--densify-until`), so each
run ends with a few hundred plain Adam steps, as the reference's (densifying until 15000 of 30000)
does: a run ending on a densification scores fresh clones and halves that never trained. `check.py` checks each run: a held-out PSNR and a fall of the loss from the
first epoch to the last, a finite PLY of exactly the splats in use, at most `--max-splats`, and
the PLY, rendered again by splat_render from each held-out camera, scoring within 0.5 dB of what
splat_train reported, which checks the readback, the PSNR, the count and the export independently.

- **perturb**: the splats perturbed (`make_synthetic.py perturb`) and fitted back for 2000
  iterations from a capacity of 256 entries, which the first iterations overflow, with
  `--reset-interval 900` (a reset at 900, size prunes after it) and `--max-splats 4096`,
  which densification fills by iteration 800, so later additions are dropped (the overflow
  path). It requires a PSNR within 0.5 dB of the native builds' (35.9: 35.94 on lavapipe, 35.91
  on MoltenVK, from 20 at the start; test.sh's `native_perturb_psnr`, which a native build may
  exceed by 1 dB at most, so it's kept current), which makes it the web's parity check
  ([web.md](web.md#training-on-the-web)), a loss falling 8
  times (34.0) and at least one skipped iteration (8: covering Adam's skip, Record's −1 and the
  report).
- **sparse**: 200 splats initialized from a tenth of the positions as the converter initializes
  COLMAP points (`make_synthetic.py sparse`), trained for 1800 iterations with `--max-splats
  12288` and SH degree 0. Densification must grow the count (`check.py --grow`: to 11470 of the 12288
  on lavapipe) and the held-out PSNR reach 28 dB (37.0 on MoltenVK). One run's PSNR swings by
  up to 0.7 dB with the least change to its inputs, so it isn't held to native's.
- **last**: from perturb's start, 300 iterations densifying from 100 with `--reset-interval 300`:
  iteration 200 densifies, and 300, the last, would densify and reset if it weren't (the reference
  scores and saves an iteration before its densification and reset). It requires exactly one
  densification (`check.py --densifications 1`), a held-out PSNR of 22 dB (26.7 on MoltenVK; a
  reset on the last iteration scored 13.3) and a loss falling 2 times (3.3). The 22 dB bar comes
  from MoltenVK alone; the lavapipe and web scores are recorded here after CI.

perturb and sparse train on (1 − 0.2)·L1 + 0.2·D-SSIM; with `--lambda-dssim 0` (L1 alone) they score 35.9 and
32.9 dB, the loss falling 16.3 and 8.8 times.

Each run prints its held-out PSNR and splat count (and check.py its checks), so CI's log shows
the margins on lavapipe.

The test takes about 18 s on the M4 with validation (14 s with L1 alone). From perturb's start training converges slowly: 300 iterations reach about 26 dB
(29 with a 10× `--feature-lr`: Adam moves a parameter about its learning rate a step, and perturb's
random colours are up to 3.5 off), and a degree-3 ground truth about 29 dB after 5000 (22 views don't pin the random view-dependent
colours down for the held-out ones), hence the degree-0 target and 2000 iterations.

**Cost** (`--timings`, MoltenVK on the M4, the COLMAP scene converted at 748×748: 12,900 splats
from its points, up to 910k tile entries a frame, 500 iterations; medians): `backwardRasterize`
11.2 ms against the forward `rasterize`'s 2.8 ms (4.0×), the sort 1.4 ms, `gather` 1.1 ms, Adam
0.08 ms. The backward's deterministic per-entry reduction (two barriers and a 9-value reduction
per entry up to the tile's last, against the forward's one barrier per 256) is the training
frame's largest cost, about 12 ns per entry across the GPU. Worth optimizing once densified
scenes show it dominating (a follow-up). These were measured before the chunk passes, when the graph had one
`backwardRasterize` and one `gather` node. Each pass is now a node of its own (`backwardRasterize
0`, `backwardRasterize 1`, ..., likewise `gather`), so `--timings` prints one line a pass. A
frame's backward cost is the sum of its passes' times, and the unused passes' empty dispatches
show up as near-zero lines.

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

(struct ViewPose           ; 32 bytes: the viewer's pose relative to its start camera
  (offset vec3)            ; 0: the camera centre minus the start's, in world space
  (yaw float)              ; 12: radians in about [−π, π], turning right about the start's up
  (pitch float))           ; 16: radians, looking up, within ±89°; then 12 bytes of padding

(struct DensifyStat        ; 8 bytes: a splat's densification statistics (train graph)
  (gradientNorm float)     ; 0: the sum of densifyNorm(dL/dmean) over the frames that saw it
  (visible uint))          ; 4: those frames' number
```
