# vulkan_splatting

3D Gaussian splatting (Kerbl et al., SIGGRAPH 2023) written in ZGAL, the language of the Vulkan
wrapper in `external/vulkan_test` (git submodule), for native Vulkan and WebGPU. It was
rewritten from scratch; the pre-ZGAL code is in git history before this rewrite.

Read the docs for the area you touch:

- [docs/architecture.md](docs/architecture.md): the directory layout, the pipeline, the conventions (cameras, buffers, budgets) every kernel follows, the render, train and view graphs, and each program's options and tests (splat_render, splat_view, splat_train).
- [README.md](README.md#viewing-a-scene): how to view a scene, natively and in a page, and encode a stream for the web.
- [docs/building-and-testing.md](docs/building-and-testing.md): before building, testing, adding a program or test, or bumping the wrapper.
- [docs/data-format.md](docs/data-format.md): the converter (`tools/`) and the files the programs read: the `Splat` and `Camera` buffers, the packed images, `manifest.json`, INRIA PLY, and the stream files (`lib/splat_stream.zgal`, encoded by `lib/native/`).
- [docs/web.md](docs/web.md): the programs on WebGPU: every kernel's binding and workgroup-memory budget against WebGPU's default limits, the web tests, and what the web doesn't support.
- [docs/paper-conformance.md](docs/paper-conformance.md): why the old code was replaced, and where the rewrite follows or departs from the paper; update it with any new departure.
- [docs/results.md](docs/results.md): training runs on real scenes (held-out PSNR and SSIM, time, splats) and how to reproduce them.
- ZGAL itself: `external/vulkan_test/CLAUDE.md` and the reference it points to.
