# vulkan_splatting

3D Gaussian splatting (Kerbl et al., SIGGRAPH 2023): rendering and training of splat scenes, written
in ZGAL, the language of the Vulkan wrapper in `external/vulkan_test`. Every program builds
for native Vulkan and for the web (WebGPU, under Node with Dawn or in a page); what the web does
not support is in [docs/web.md](docs/web.md#not-supported-on-the-web).

## Results

Training on a real scene, the 46-camera COLMAP capture `colmap/colmap_exports3` (748×748): 7,000
iterations with `--max-screen-size 20` reach 20.41 dB PSNR and 0.782 SSIM on the six held-out
views. The table of runs, and how to reproduce them, is in [docs/results.md](docs/results.md).

![held-out view 8, 7k iterations, --max-screen-size 20: photo (left) and render (right)](docs/results/cs5330-7k-ms20-view8.png)

## Build and test

Clone with submodules (`git clone --recursive`, or `git submodule update --init --recursive`), then
`make` builds every program into `build/`, and `make test` runs the tests. The requirements, the
other make targets, the web build and the CI are in
[docs/building-and-testing.md](docs/building-and-testing.md), and [docs/web.md](docs/web.md)
covers the web.

## Training on your own COLMAP data

`tools/convert.py colmap` turns a COLMAP text export into a scene the programs read; `splat_train`
then trains on it and writes an INRIA PLY. The converter's options, the scene's files and the
training options are in [docs/data-format.md](docs/data-format.md) and the
[splat_train section](docs/architecture.md#splat_train) of the architecture doc. A worked example,
from conversion through `tools/scene_args.py --train` to the evaluation, is in
[docs/results.md](docs/results.md#reproducing).

## Viewing a scene

`splat_view` draws a scene while it streams in: natively an INRIA PLY (as splat_train writes) or a
`scene.stream`, the compact form the web reads. Every option is in the
[splat_view section](docs/architecture.md#splat_view) of the architecture doc.

### Encoding a stream

`tools/convert.py stream` encodes a PLY (or a converted scene's `splats.bin`) as `scene.stream` in
an output directory, with a preset from `tools/stream-configs/`: `full` (every value f32, drawn
exactly as the PLY), `f16` or `web`, the smallest (80 bytes a splat at SH degree 3 against
`full`'s 236). `--cameras` copies a `cameras.bin` beside it. It calls the C++ encoder through the
native library, so build that first (`make native`, which `make` does too):

```
tools/convert.py stream scene.ply out --config tools/stream-configs/web.json --cameras cameras.bin
```

The presets and their errors are in [docs/data-format.md](docs/data-format.md#stream-files).

### Natively

```
build/splat_view/splat_view scene.ply
build/splat_view/splat_view out/scene.stream --cameras out/cameras.bin --camera 3
build/splat_view/splat_view out/scene.stream --frames 16 --capture view.png --size 640x480
```

It opens an 800×800 window (`--size WxH` for another) at the camera the encoder wrote into the
stream's header, or at camera k of a `cameras.bin` with `--cameras FILE --camera k`, and the view
fills in as the file loads (`--stream-words`, 4 MiB a frame by default). `--frames N` runs N frames
headless instead, and `--capture FILE` writes the last one to a PNG.

| Input | Does |
| --- | --- |
| W, A, S, D | move forward, left, back, right |
| Q, E | sink, rise |
| drag (left mouse button) | look around |
| scroll | move faster (up) or slower (down) |
| Escape | quit |

The speed starts at a quarter of the cameras' spread a second with `--cameras`; without it, or when
the file's cameras don't spread (one camera, or coincident ones), of the start camera's distance
from the world origin, which suits only a scene around the origin, so scroll when it's off.

### On the web

`make web` builds every program for WebGPU (`make build/web/splat_view` the viewer alone). The
page, `splat_view.zgal.html`, takes its arguments from the query string (`arg=` for the scene,
`name=value` for the options) and fetches the files by URL, so serve the build and the scene
together, e.g. from `build/site`:

```
mkdir -p build/site/scene
ln -s ../web/splat_view build/site/viewer
cp out/scene.stream out/cameras.bin build/site/scene/
python3 -m http.server 8000 --bind 127.0.0.1 --directory build/site
```

then open
`http://127.0.0.1:8000/viewer/splat_view.zgal.html?arg=/scene/scene.stream&cameras=/scene/cameras.bin&camera=0`
in a browser with WebGPU. A relative path resolves against the page's `assets/` directory, so give
the scene's path from the server's root. WebGPU needs a secure context, which `127.0.0.1` is.
Under Node the same build runs headless only:

```
node build/web/splat_view/splat_view.zgal.main.js out/scene.stream --frames 16 --capture view.png
```

The web reads streams only: a PLY is refused as a stream whose header is bad (code 1), so convert
it first. The rest of what the web doesn't support is in
[docs/web.md](docs/web.md#not-supported-on-the-web).

## Layout

The fuller version is [docs/architecture.md](docs/architecture.md#layout).

- `programs/`: one directory per program (`splat_render`, `splat_view`, `splat_train`) and per GPU test.
- `lib/`: the ZGAL modules the programs share (the render, view and train graphs, the splat math, SSIM, the stream format).
- `tools/`: the offline Python converter, evaluator and synthetic scene generator, with tests.
- `tests/`: C++ tests and closed-form oracles.
- `colmap/`: the committed COLMAP scene and its images.
- `external/vulkan_test`: the wrapper and ZGAL.

## Docs

- [docs/architecture.md](docs/architecture.md): the layout, the render and train pipelines, the view graph, and the conventions every kernel follows.
- [docs/building-and-testing.md](docs/building-and-testing.md): building, testing, adding programs and tests, CI.
- [docs/data-format.md](docs/data-format.md): the converter, the scene files, the INRIA PLY format and the stream files.
- [docs/web.md](docs/web.md): the WebGPU build, its limits and tests.
- [docs/paper-conformance.md](docs/paper-conformance.md): where the code follows the paper and where it departs.
- [docs/results.md](docs/results.md): training runs on real scenes.
- [CLAUDE.md](CLAUDE.md): an index for agents working in the repository.
