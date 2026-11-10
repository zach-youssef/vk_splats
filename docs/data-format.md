# Data format

The programs read scenes that `tools/convert.py` (or the synthetic scene generator) writes
offline: std430 `buffer-file`s and a `manifest.json`, the same on native and the web. The code is
the `tools/splatdata` package; its tests are `tools/tests/`.

## The converter

```
tools/convert.py colmap <model dir> <out dir> [--images <dir>] [--max-size <px>] [--splats <ply>]
tools/convert.py ply-to-bin <in.ply> <out splats.bin>
tools/convert.py bin-to-ply <in splats.bin> <out.ply>
tools/convert.py stream <in.ply|splats.bin> <out dir> --config <file> [--cameras <cameras.bin>]
```

`stream` writes a [stream file](#stream-files) instead of a scene.

For example, the real scene:
`tools/convert.py colmap colmap/colmap_exports3 build/data/cs5330 --images colmap/cs5330_dense_resized`
(about 15 s).

Reading a PLY (`ply-to-bin`, `colmap --splats`, `stream`, `evaluate.py`) and encoding a stream go
through the native library,
so build it first: `make native` ([building and testing](building-and-testing.md#the-native-library)).

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
  (imageBatch uint)        ; 104: which images_<k>.bin; 0xFFFFFFFF without an image
  (heldOut uint))          ; 108: 1 for a test camera (the manifest's split), else 0
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
 "version": 2,
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
- `version` changes with any change to the files' layout. Version 2 added the Camera's `heldOut`
  (the split, for the trainer, which can't read the manifest), in what was padding.

## INRIA PLY

This is the reference code's format, read and written by `splatdata.splats`: `read_ply` calls the
C++ reader the programs share (`lib/native/ply_reader.h`, through `libsplat_native`,
[building-and-testing.md](building-and-testing.md#the-native-library); splat_view's native opener
reads a PLY with it and encodes it in full, f32, into a stream in memory), while `write_ply` is
still Python, the one remaining duplicate of C++ (splat_train's export), until that writer moves
to `lib/native/` too. It is binary
little-endian, with one `vertex` element of float properties:

```
x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3
```

- `f_rest` is channel-major: coefficient k (1–15) of channel c is `f_rest_{c·15 + k − 1}`.
- The scales are logs, the opacity is a logit, and `rot` is (w, x, y, z).

The reader accepts properties in any order and of any numeric type (each converted to double,
then float), SH of degree 0–2 (fewer `f_rest`, padded with zeros; it reports the degree) and
elements after the vertices. It ignores the normals. It refuses anything else (another format, a
list or duplicate vertex property, missing properties, `f_rest` not numbered from 0, fewer
vertices than declared) with a message. The writer
writes degree 3, with zero normals.

## Stream files

A stream is one little-endian file of splats that a viewer can draw while it downloads:

```
[StreamHeader (432 B)][base record × splatCount][SH record × splatCount]
```

Once the base records have arrived every splat can be drawn (at degree 0); the SH records add the
higher bands as they follow. Each field group has its own encoding, given in the header, so
changing the precision needs no rebuild. The format, `checkStreamHeader` and the decode are
`lib/splat_stream.zgal` (so the shaders, the C++ and the TypeScript run the same code); the
encoder is `lib/native/stream_encoder.h`. `tests/cpp/splat_stream_test.cpp` checks them,
`tests/cpp/default_camera_test.cpp` the default camera, `tools/tests/test_stream.py` the
converter, and `programs/stream_test` the [decode](#decode) on the GPU. `programs/splat_view`
draws a stream ([architecture.md](architecture.md#splat_view)).

### Converter

```
tools/convert.py stream <in.ply|splats.bin> <out dir> --config <file> [--cameras <cameras.bin>]
```

reads the splats (an INRIA PLY, `.ply`, or a splats.bin, `.bin`, through the native library's
readers), encodes them with the C++ encoder as the JSON config says, and writes
`<out dir>/scene.stream`; with `--cameras` it checks that the file is whole `Camera`s and copies it
to `<out dir>/cameras.bin`. The outputs are written to temporary files and renamed once all are
written, so a failure leaves neither a partial file nor a stream without its cameras. A PLY and the
splats.bin of the same splats give the same bytes when the PLY is of degree 3 (as `write_ply` and
splat_train write them) or the config gives `shDegree` as a number: a splats.bin is always of
degree 3, so under `"data"` a lower-degree PLY gives a lower-degree stream. Then
it prints the layout and, per group, the encoding, the largest error (raw and activated, as the
[encoder](#encoder) reports them) and the range the header holds (low to low + stepSize ·
(2^bits − 1)). The web preset on the tests' fixture (`stream_fixture.py out --count 40 --seed 1`):

```
out/scene.stream: 40 splats, SH degree 3, 80 B per splat (base 32, SH 48): 3632 B, the SH records from byte 1712
  group     encoding  max error  activated  (of)         range
  position  f32               0          0  value        -
  scale     f16         0.00193   0.000153  exp          -
  rotation  u8          0.00392     0.0136  angle (rad)  explicit [-1, 1]
  opacity   u8           0.0756      0.017  sigmoid      data [-20, 20]
  dc        f16         0.00025    0.00025  value        -
  sh        u8          0.00572    0.00572  value        data [-1.42, 1.501]
```

The config gives the SH degree and every group, with no defaults, since each choice trades size
for precision (`tools/splatdata/stream.py` checks it, naming the file and key of any problem):

```json
{"shDegree": "data",
 "position": {"encoding": "f32"},
 "scale": {"encoding": "f16"},
 "rotation": {"encoding": "u8", "range": [-1, 1]},
 "opacity": {"encoding": "u8", "range": "data"},
 "dc": {"encoding": "f16"},
 "sh": {"encoding": "u8", "range": "data"}}
```

- `shDegree`: 0–3, or `"data"`, the input's degree (a splats.bin's is always 3; a PLY's is its
  `f_rest` count's). A degree above the input's is refused.
- The groups, `position`, `scale`, `rotation`, `opacity`, `dc` and `sh`, each take an `encoding`:
  `f32`, `f16`, `u16` or `u8`. A quantized one (`u16`, `u8`) needs a `range`, which f32 and f16
  don't take: `"data"` (the data's minimum and maximum, per component, or over every component
  for SH), `[lo, hi]` for every component, or one `[lo, hi]` per component (not for SH, which
  shares one range). Values outside an explicit range are clamped; `lo` ≤ `hi`, finite in
  float32.
- Unknown keys are refused.

The presets in the table below are `tools/stream-configs/{full,f16,web}.json`, all at
`"shDegree": "data"`. The encoder's refusals (a non-finite value, a zero quantized rotation, an f16
overflow, too many splats, a range too wide for a float32 step) are printed with the group and
splat, prefixed with the input's path, or the config's for an explicit range, and nothing is
written; so are a bad config, an unreadable input and a failed write.
`tools/tests/stream_fixture.py <out dir> [--count <n>] [--seed <n>]` writes the tests' fixture,
the same splats as `fixture.ply` and `splats.bin`, with the values the encodings have edges at.

### Header

std430, no padding; offsets in bytes:

| Offset | Field | |
|---|---|---|
| 0 | `magic` | `"SPLS"` (`streamMagic`, 1397510227 as a little-endian uint) |
| 4 | `version` | 1 |
| 8 | `splatCount` | at most 559,240 (`streamMaxSplats`: the f32 `Splat` buffer's 128 MiB) |
| 12 | `shDegree` | 0–3: the SH record holds bands 1 to `shDegree` |
| 16 | `baseStride` | bytes per base record, a multiple of 4, at most 56 |
| 20 | `shStride` | bytes per SH record, a multiple of 4, at most 180, and 0 at degree 0 |
| 24 | `baseOffset` | 432 |
| 28 | `shOffset` | `baseOffset + splatCount · baseStride`; the file ends at `shOffset + splatCount · shStride` |
| 32 | `camera` | a `Camera` (112 B): the default camera, below |
| 144 | `fields[6]` | one `StreamField` (48 B) per group |

The groups, in `fields` order, with `Splat`'s components:

| Group | Record | Components |
|---|---|---|
| 0 position | base | xyz |
| 1 scale | base | log-scale xyz |
| 2 rotation | base | (w, x, y, z) |
| 3 opacity | base | the logit |
| 4 DC | base | `sh[0..2]` |
| 5 SH | SH | `sh[3..]`: 3·((d+1)² − 1), `sh[3k + c]` order |

A `StreamField` is `encoding`, `components`, `offset` (the first component's byte within its
record; the components follow it, `encodingSize` bytes each) and `sharedRange` (uints), then
`low` and `stepSize` (vec4s). The encodings:

| Value | Encoding | Bytes | Decode |
|---|---|---|---|
| 0 | f32 | 4 | the bits |
| 1 | f16 | 2 | IEEE half (the encoder rounds to nearest, ties to even) |
| 2 | u16 | 2 | `low + stepSize · q` in float32 |
| 3 | u8 | 1 | `low + stepSize · q` in float32 |

A quantized component's range is `low[c]` and `stepSize[c]`, or `.x` of both for every component
when `sharedRange` is 1 (required for SH, whose components outnumber a vec4's). The encoder
computes `stepSize = (high − low) / (2^bits − 1)` in float32 and stores it, so a decode needn't
redo the division; `low` and `stepSize` are zero for f32 and f16. A component sits within one
little-endian word (offsets are aligned to the component's size and records start on words), so
`decodeComponent` decodes it from that word and its byte offset.

`checkStreamHeader` returns `streamOk` (0) or the first problem: 1 magic, 2 version, 3 SH degree,
4 too many splats, 5 a stride not a multiple of 4, above its maximum, or a nonzero `shStride` at
degree 0, 6 `baseOffset` or
`shOffset` not where they must be, 7 an encoding, 8 a group's component count (for the SH group,
at `shDegree`), 9 `sharedRange` (not 0 or 1, or 0 with more than four components), 10 a field
misaligned, past its record's stride, or overlapping another base field. It doesn't check the
file's size, which a stream doesn't know up front: the reader fails on an early end or extra
bytes. A stride may be larger than its fields need (the rest is padding). Each field's offset is
bounded by its stride before its end is computed, so with the stride caps no offset computed from
a checked header overflows a uint, and the largest stream (131,981,072 B) fits one 128 MiB
binding.

### Decode

Three kernels in `lib/splat_stream.zgal` decode a stream into the f32 `Splat` working buffer as its
records arrive, each record once, so `projectSplat` reads what it always reads. The graph binds the
stream as a buffer of its words (`uint`s) and gives PlanDecode `arrived`, the bytes in that buffer
so far (a multiple of 4, at most the buffer's), and `capacity`, the working buffer's splats. All
four of their buffers are `#:frames shared`, since the decode is incremental:

- `splats`: the working buffer, at least `splatCount` elements, zeroed at creation, so a splat whose
  SH record hasn't arrived evaluates as degree 0 and the SH above the stream's degree stay 0;
- `header`: one `StreamHeader`, PlanDecode's copy of the stream's once `arrived` covers it (whether
  or not it passes), which the decode kernels read and an app can read back (its camera);
- `counts`: `uint`s, `decodeCountsSize` (7) at least, zeroed: the records decoded
  (`decodeBaseDecoded`, `decodeShDecoded`: splats [0, n) have them), this frame's ranges
  (from `decodeBaseFrom`, `decodeShFrom`), this frame's `arrived` (`decodeArrived`),
  `decodeHeaderState` (0 until the header is in, then 1) and `decodeHeaderCode`;
- `commands`: two `DispatchCommand`s (`(import indirect)`), DecodeBase's and DecodeSh's.

Each frame:

1. **PlanDecode** (one invocation) writes `arrived` and zero-workgroup commands. Once `arrived` ≥ 432
   it reads the header from the stream's words, copies it to `header` (the first time), and writes
   `checkStreamHeader`'s code, or `streamOverCapacity` (11) for a header that passes with
   `splatCount` above `capacity`. Only a code of 0 decodes: the records now complete are
   `completeRecords(arrived, offset, stride, splatCount)` for the base and the SH records (whole
   records only; all of them for the SH's stride 0 at degree 0, which hold nothing), never fewer
   than before, and the commands cover the new ones, 64 a workgroup (`decodeGroups`; at most 8,739).
2. **DecodeBase**, dispatched indirectly over the new base records, decodes each component with
   `decodeComponent` from the stream's word holding it, the descriptors from `header` (so a precision
   change needs no rebuild), writing position, logScale, rotation, opacity and `sh[0..2]` field by
   field, leaving `sh[3..]`.
3. **DecodeSh** likewise writes the new SH records' `sh[3..]` (none at degree 0).

The kernels take the stream's size on trust: whatever supplies `arrived` (the app in stream_test,
from its stream-buffer's `streamed` events in splat_view) keeps it within the buffer. `programs/stream_test` tests them
([building-and-testing.md](building-and-testing.md#stream-decode-check)).

### Encoder

`encodeStream(splats, config, out, report)` takes an `EncodeConfig`: `shDegree` and, per group,
the encoding and how a quantized range is chosen: `kRangeData` (the data's minimum and maximum, per
component, or over every component for SH) or `kRangeExplicit` (`lo`, `hi`, per component or
`[0]` for SH; values outside are clamped). f32 and f16 groups must say `kRangeNone`. A quantized
rotation is normalized first; an f32 one is copied as it is. Each value is encoded to the code
whose float32 decode is nearest.

`streamLayout` gives the header's layout: base fields are placed largest component first (group
order within a size), so only the stride's rounding to 4 pads a record, and padding bytes are
zero, so the bytes are deterministic. The presets' records (`tools/stream-configs/`):

| Preset | Position | Scale | Rotation | Opacity | DC | SH | Base / SH stride |
|---|---|---|---|---|---|---|---|
| full | f32 | f32 | f32 | f32 | f32 | f32 | 56 / 180 |
| f16 | f32 | f16 | f16 | f16 | f16 | f16 | 36 / 92 |
| web | f32 | f16 | u8, [−1, 1] | u8, data | f16 | u8, data | 32 / 48 |

The report gives each group's largest error over every splat and component, raw and activated
(exp of the scale, sigmoid of the opacity, the angle in radians between the rotations, the value
itself otherwise; an exp overflow that changes the value counts as infinite), decoding through
`decodeComponent`, so it measures what the shaders read. The entry points (and `defaultCamera`)
take plain structs and spans and return a status, never throwing. `encodeStream`'s statuses: a bad
config (with the group; `groupCount` for `shDegree`), too many splats, an output of the wrong size,
a range too wide for a float step (with the group), out of memory, and, with the group and the
splat, a non-finite input value, a zero quantized rotation or an f16 overflow (|v| ≥ 65520).
Values the stream drops (SH above `shDegree`) aren't checked.

### Default camera

The encoder writes a default camera into the header (`lib/native/default_camera.h`), so a viewer
opening a scene without cameras still sees it. Per axis, the order statistics at
`floor(p (N − 1))` for p = 0.05 and 0.95 bound a box (robust to a few floaters), with centre c and
radius r, half its diagonal (1 for a point; an empty scene has c = 0, r = 1). The camera is
1280×720 with a 60° vertical field of view (fx = fy = 360√3, principal point at the centre),
looks down world +z with the identity rotation from c − 2.2 r e_z: r / sin 30° puts the box's
bounding sphere inside the vertical field of view, and the factor 1.1 adds a margin. Everything is
float32 with literal constants, so the camera is the same on every platform and for any splat
order.
