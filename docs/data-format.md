# Data format

The programs read scenes that `tools/convert.py` (or the synthetic scene generator) writes
offline: std430 `buffer-file`s and a `manifest.json`, the same on native and the web. The code is
the `tools/splatdata` package; its tests are `tools/tests/`.

## The converter

```
tools/convert.py colmap <model dir> <out dir> [--images <dir>] [--max-size <px>] [--splats <ply>]
tools/convert.py ply-to-bin <in.ply> <out splats.bin>
tools/convert.py bin-to-ply <in splats.bin> <out.ply>
```

For example, the real scene:
`tools/convert.py colmap colmap/colmap_exports3 build/data/cs5330 --images colmap/cs5330_dense_resized`
(about 15 s).

`colmap` reads a COLMAP text export (`cameras.txt`, `images.txt`, `points3D.txt`; other files are
ignored) and writes a scene:

- **Cameras**: one per image, sorted by image name, each undistorted to a pinhole camera.
- **Images** (with `--images`, named as in `images.txt`): undistorted, then resized with Lanczos
  so the longer side is at most `--max-size` (aspect ratio kept).
- **Splats**: initialized from every point in `points3D.txt`, or read from an INRIA PLY with
  `--splats`.
- **Split**: every 8th camera in name order, starting with the first, is held out for testing (the
  reference code's `llffhold`).

Parsing follows COLMAP's writer: `images.txt` has two lines per image (the second, POINTS2D, may
be empty), `points3D.txt` lines have tracks of any length, quaternions are stored w first, and
CRLF line endings are accepted. Camera models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL and
OPENCV, with COLMAP's distortion formulas.

### Undistortion

Each output camera keeps its source's size and principal point, and its focal lengths are scaled
by the smallest factor that keeps every output pixel inside the distorted image (so no pixel is
blank or extrapolated; about +2% for cs5330's SIMPLE_RADIAL camera) and over which the distortion
doesn't fold (its Jacobian stays positive, so strong barrel distortion can't wrap the corners back
inside the image); the converter rejects a camera for which no such factor exists. Pixels are
resampled bilinearly at the distorted position of each pixel's pinhole ray. Cameras without
distortion are copied unchanged. Every image must have its camera's size (the converter rejects
others rather than stretching them). Resizing scales the intrinsics by width'/width and height'/height; both keep
COLMAP's pixel convention, where the top-left pixel's centre is (0.5, 0.5).

### Initialization

As in the paper (Sec. 5.1) and the reference code:

- **Positions**: the points.
- **Scales**: isotropic. The log-scale is log √(mean of the squared distances to the 3 nearest
  other points), with the mean clamped to at least 1e-7, as simple-knn's `distCUDA2` computes it.
- **Rotations**: identity.
- **Opacity**: 0.1, stored as its logit.
- **SH**: the point's colour in the DC term, `(rgb − 0.5) / C0` with C0 = 0.28209479177387814;
  the other coefficients are zero.

## Output directory

| File | Contents |
|---|---|
| `manifest.json` | the scene's description (below) |
| `splats.bin` | `Splat` structs |
| `cameras.bin` | `Camera` structs, in manifest order |
| `images_<k>.bin` | the images, RGBA8, batch k |
| `images/<name>.png` | each undistorted image, as packed (for viewing and PSNR) |

### Splat (240 bytes)

```
(struct Splat
  (position vec3)          ; offset 0
  (opacity float)          ; 12: logit
  (logScale vec3)          ; 16: then 4 bytes of std430 padding
  (rotation vec4)          ; 32: quaternion (w, x, y, z) in (.x, .y, .z, .w), not normalized
  (sh (array float 48)))   ; 48: coefficient k (0-15) of channel c (RGB) at sh[3k + c]
```

The values are pre-activation: the kernels apply exp to the scale, sigmoid to the opacity, and
normalize the quaternion. The colour is `max(Σ_k sh_k·Y_k(dir) + 0.5, 0)`, as in the reference
code.

### Camera (112 bytes)

```
(struct Camera
  (worldToCamera mat4)     ; offset 0: column-major; [R t; 0 1], X_cam = R X + t
  (intrinsics vec4)        ; 64: (fx, fy, cx, cy) in pixels
  (center vec3)            ; 80: the camera centre in world space, −Rᵀt
  (pixelOffset uint)       ; 92: the image's first pixel in its batch, in pixels
  (size uvec2)             ; 96: (width, height)
  (imageBatch uint))       ; 104: which images_<k>.bin; 0xFFFFFFFF without an image
                           ; then 4 bytes of padding
```

The convention is COLMAP's: +z looks forward, +y down, and pixel (x, y) projects at
`(fx·X/Z + cx, fy·Y/Z + cy)`, with pixel centres at integer + 0.5. The cameras are pinhole (no
distortion).

### Images

Each pixel is a `uint`: R in the low byte, then G, B and alpha (255), so the bytes are R, G, B, A.
Rows run top to bottom, and the images are packed in camera order. A batch holds whole images and
at most 128 MiB (the largest buffer any target binds), so a new batch starts when the next image
doesn't fit. cs5330 at 748² fits in one batch (103 MB). The values are the images' 8-bit sRGB
values, used as stored (the reference code trains on them without linearizing).

### manifest.json

```
{
 "version": 1,
 "splatCount": 12900,
 "shDegree": 3,
 "cameraCount": 46,
 "extent": 6.53,
 "files": {"splats": "splats.bin", "cameras": "cameras.bin", "images": ["images_0.bin"]},
 "train": [1, 2, ...],
 "test": [0, 8, ...],
 "cameras": [{"index": 0, "name": "...jpg", "split": "test", "width": 748, "height": 748,
              "fx": ..., "fy": ..., "cx": ..., "cy": ...,
              "imageBatch": 0, "pixelOffset": 0, "png": "images/....png"}, ...],
 "source": {"colmap": "colmap_exports3", "splats": "points3D", "maxSize": null}
}
```

- `train` and `test` are camera indices.
- `extent` is the scene extent that the reference code scales the position learning rate by: 1.1
  times the largest distance from a camera centre to the cameras' mean centre (`getNerfppNorm`),
  over the training cameras only, as in the reference (all cameras when none is a training one).
- A camera without an image has no `imageBatch`, `pixelOffset` or `png`.
- `version` changes with any change to the files' layout.

## INRIA PLY

This is the reference code's format, read and written by `splatdata.splats`. It is binary
little-endian, with one `vertex` element of float properties:

```
x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3
```

- `f_rest` is channel-major: coefficient k (1–15) of channel c is `f_rest_{c·15 + k − 1}`.
- The scales are logs, the opacity is a logit, and `rot` is (w, x, y, z).

The reader accepts properties in any order and of any numeric type, SH of degree 0–2 (fewer
`f_rest`, padded with zeros) and elements after the vertices. It ignores the normals. The writer
writes degree 3, with zero normals.
