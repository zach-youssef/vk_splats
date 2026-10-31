# Architecture

## Layout

The planned layout, filled in as the M1 and M2 tasks land.

- `programs/<name>/`: one ZGAL program each (`<name>.zgal` with a `main`, plus its own `.cpp` and
  `.ts` files, `assets/` and optional `test.sh` / `test-web.sh`). The Makefile finds them; adding
  one needs no Makefile change ([building-and-testing.md](building-and-testing.md)).
- `lib/`: ZGAL modules the programs share, imported by relative path:
  `(import "../../lib/<module>.zgal")`. A module's name is a C++ namespace (ZGAL compiles its
  functions to C++ too), so it uses underscores, not hyphens.
  - `splat_math.zgal`: the structs, the per-splat preprocess (covariance, EWA projection, conic,
    radius, tile rectangle, SH colour) and the sort keys ([below](#splat-math)).
  - `render_graph.zgal`: `SplatGraph`, the render graph's preprocess, tile entries, sort and
    tile ranges ([below](#render-graph)); it imports `splat_math.zgal`.
- `tools/`: offline Python (numpy, scipy, Pillow): the converter from COLMAP text and INRIA PLY files to the
  programs' inputs, and the synthetic scene generator.
- `tests/`: C++ tests (`tests/cpp/`), closed-form oracles (`tests/oracle/`), Python tests and test
  data.
- `colmap/`: the one real scene: `colmap_exports3`, a COLMAP text export (one SIMPLE_RADIAL
  camera, 46 images, sparse points; CRLF line endings), and its images, `cs5330_dense_resized`
  (748×748 JPEGs, named as in `images.txt`), committed directly (6.2 MB, no LFS).
- `external/vulkan_test`: the wrapper and ZGAL.

## Pipeline

The planned pipeline (M1 renders, M2 trains). Steps 1 to 4 are `lib/render_graph.zgal`
([below](#render-graph)); the rasterizer and training are to come.

Data is converted offline: the programs read std430 `buffer-file` data and a `manifest.json`, the
same on native and the web, so no image or COLMAP parsing library is linked. The training images
are undistorted and packed by the converter into an RGBA8 buffer-file (one `uint` per pixel), not
bound as images: ZGAL binds only fixed `image-file` lists as arrays, and `import-images` is native
only.

Per frame, rendering will follow the paper's tile-based rasterizer:

1. **Preprocess** (per splat): 3D covariance from scale and rotation, EWA projection to a 2D conic
   (plus 0.3 on the diagonal), a 3σ radius, near-plane and off-screen culling, the tile rectangle,
   and the SH colour for the view direction.
2. **Tile entries**: an exclusive scan of the per-splat tile counts sizes the entry buffer; each
   splat writes one 32-bit key (tile, then quantized depth) per tile it covers.
3. **Sort** the keys with their splat indices, once, globally.
4. **Tile ranges**: each tile's range of entries, from comparing neighbouring keys.
5. **Rasterize**: one 16×16 workgroup per tile blends its range front to back, in shared-memory
   batches, writing the colour, the final transmittance and the last contributor (kept for the
   backward pass).

Training will add the backward pass (the reverse traversal, then the preprocess chain), Adam, and
densification and pruning on a fixed capacity, compacted by scan.

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
  buffer, e.g. the images per batch.
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

`lib/render_graph.zgal`'s `SplatGraph` runs pipeline steps 1 to 4 for one camera per frame; the
rasterizer (to come) reads its `ranges`, `entrySplats` and `projected`.

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
   neighbouring keys' tiles differ. Tiles without entries keep (0, 0).

`counts` holds (entries, total, capacity, depth bits, depth shift) and is read back every frame.

- **Dispatch sizes**: a dispatch has at most `maxGroups` workgroups (a required graph parameter):
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
  or 3. `tests/cpp/render_graph_test.cpp` checks `groupsFor` and `grownCapacity` on the CPU. The
  preprocess itself is compared to float precision (mean 1e-4 relative, depth 1e-5, radius ±1),
  since the GPU's `exp`, `sqrt` and division needn't round as C++'s do. Degenerate splats (a zero
  quaternion, logScale 100 and 30, appended by `scenes.py`; `--culled-tail`) must be culled on both;
  the `!(det > 0)` NaN guard holds on MoltenVK.
- **Test inputs**: ZGAL has no expression for a `buffer-file`'s element count and no path
  joining, so tile_test takes the files, splat count and camera size as options; its `test.sh`
  reads them from the scene's `manifest.json` (`scenes.py`), and the check verifies them against
  the readbacks. The test also runs with `--timings` (a frame waits for the GPU) so that each
  frame's readbacks arrive: a vulkan_test gap, to remove after its readback-delivery fix lands.

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
  (imageBatch uint))       ; 104: which images_<k>.bin; 0xFFFFFFFF without an image
                           ; then 4 bytes of padding

(struct Projected          ; 64 bytes: one splat as one camera sees it (the preprocess's output)
  (mean vec2)              ; 0: the 2D mean, in pixels
  (depth float)            ; 8: camera-space z
  (radius uint)            ; 12: the 3σ radius, in pixels
  (conic vec3)             ; 16: Σ'⁻¹ as (a, b, c) for [[a, b], [b, c]]
  (opacity float)          ; 28: sigmoid(Splat.opacity)
  (colour vec3)            ; 32: the SH colour for this camera's view direction
  (tileCount uint)         ; 44: (maxX − minX)(maxY − minY); 0 when culled
  (tiles uvec4))           ; 48: the tile rectangle (minX, minY, maxX, maxY), max exclusive
```
