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
  - the render graph (to come).
- `tools/`: offline Python (numpy, scipy, Pillow): the converter from COLMAP text and INRIA PLY files to the
  programs' inputs, and the synthetic scene generator.
- `tests/`: C++ tests (`tests/cpp/`), closed-form oracles (`tests/oracle/`), Python tests and test
  data.
- `colmap/`: the one real scene: `colmap_exports3`, a COLMAP text export (one SIMPLE_RADIAL
  camera, 46 images, sparse points; CRLF line endings), and its images, `cs5330_dense_resized`
  (748×748 JPEGs, named as in `images.txt`), committed directly (6.2 MB, no LFS).
- `external/vulkan_test`: the wrapper and ZGAL.

## Pipeline

The planned pipeline (M1 renders, M2 trains); nothing of it is implemented yet.

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
  grow.

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
  tile entry, the tile index in the high `tileBits` = ⌈log₂ tileCount⌉ (1 to 31) bits and the
  depth in the low `depthBits` = 32 − tileBits, quantized linearly in log z between `near` and
  `far` and clamped there. 748×748 has 2209 tiles: 12 tile bits, 20 depth bits. `depthKey`
  requires 0 < near < far and tile < 2^tileBits, unchecked. Keys tie for depths within one
  quantization step and for all depths ≥ far, so their order within a tile is the sort's choice:
  the tile-entry pass must not rely on the sort's stability for determinism. It writes each
  splat's entries at its scan offset (positions fixed by splat index, not by thread timing), and
  either the sort is stable or it orders ties by the entry's value (the splat index), so the
  blend order is the same on every run.

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
