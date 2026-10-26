# vulkan_splatting

3D Gaussian splatting (Kerbl et al., SIGGRAPH 2023) written in ZGAL, the language of the Vulkan
wrapper in `external/vulkan_test` (git submodule), for native Vulkan and WebGPU. It is being
rewritten from scratch; the pre-ZGAL code is in git history before this rewrite.

Read the docs for the area you touch:

- [docs/architecture.md](docs/architecture.md): the directory layout, the pipeline and the conventions (cameras, buffers, budgets) every kernel follows.
- [docs/building-and-testing.md](docs/building-and-testing.md): before building, testing, adding a program or test, or bumping the wrapper.
- [docs/data-format.md](docs/data-format.md): the converter (`tools/`) and the files the programs read: the `Splat` and `Camera` buffers, the packed images, `manifest.json`, INRIA PLY.
- [docs/paper-conformance.md](docs/paper-conformance.md): why the old code was replaced, and where the rewrite follows or departs from the paper; update it with any new departure.
- ZGAL itself: `external/vulkan_test/CLAUDE.md` and the reference it points to.
