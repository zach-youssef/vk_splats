# Architecture

## Layout

The planned layout: `programs/`, `lib/`, `tools/` and `tests/` don't exist yet and are filled in as
the M1 and M2 tasks land.

- `programs/<name>/`: one ZGAL program each (`<name>.zgal` with a `main`, plus its own `.cpp` and
  `.ts` files, `assets/` and optional `test.sh` / `test-web.sh`). The Makefile finds them; adding
  one needs no Makefile change ([building-and-testing.md](building-and-testing.md)).
- `lib/`: ZGAL modules the programs share (the splat math, the render graph), imported by relative
  path: `(import "../../lib/<module>.zgal")`.
- `tools/`: offline Python (numpy, scipy): the converter from COLMAP text and INRIA PLY files to the
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
  default limit). Struct layouts are listed here as they're added. No single buffer binds more
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
