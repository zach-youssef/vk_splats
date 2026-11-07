# The web

Every program is also built for WebGPU (`make web`, ZGAL's `zgal-web-program`) and runs under
Node with Dawn (`node build/web/<name>/<name>.zgal.main.js <options>`) or in a page
(`<name>.zgal.html`). The ZGAL side is in `external/vulkan_test/docs/web.md`; this file has what is
specific to the splatting programs.

## Limits

ZGAL's web runtime requests no limits (`requestDevice` with features only), so every kernel must
fit WebGPU's defaults, the same budgets [architecture.md](architecture.md#conventions) sets for
native. Measured from the generated WGSL (`build/web/splat_render/src/*.wgsl`, and the variants
without subgroups embedded in `splat_render.zgal.ts`):

| Kernel | Storage buffers (max 8) | Uniforms (max 12) | Storage textures (max 4) | Workgroup memory (max 16384 B) | Invocations (max 256) |
|---|---|---|---|---|---|
| InitSplatCounts | 2 | 1 | 0 | 0 | 1 |
| Preprocess | 5 | 1 | 0 | 0 | 64 |
| PlanEntries | 2 | 1 | 0 | 0 | 1 |
| EmitEntries | 6 | 1 | 0 | 0 | 256 |
| TileRanges | 3 | 1 | 0 | 0 | 256 |
| Rasterize | 5 | 1 | 1 (`rgba8unorm`) | 10240 B (256 × (8 + 16 + 16): a batch's means, conics and opacities, colours) | 16 × 16 |
| Loss | 4 | 1 | 0 | 0 | 256 |
| SsimRows, SsimGradRows, SsimGradColumns | 6 | 1 | 0 | 0 | 256 |
| SsimColumns | 8 (at the limit) | 1 | 0 | 0 | 256 |
| TileLasts | 3 | 1 | 0 | 0 | 64 |
| BackwardRasterize | 7 | 1 | 0 | 14912 B (128 × (8 + 16 + 16) for a batch, 9 × 256 × 4 and 9 × 16 × 4 for the reduction) | 16 × 16 |
| InvertEntries | 6 | 1 | 0 | 0 | 256 |
| PlanChunks | 4 | 1 | 0 | 0 | 1 |
| GatherGradients | 7 | 1 | 0 | 0 | 64 |
| BackwardPreprocess | 7 | 1 | 0 | 0 | 64 |
| Adam | 6 | 1 | 0 | 0 | 64 |
| Record | 3 | 1 | 0 | 0 | 1 |
| DensifyPrepare, DensifyFinish | 2, 3 | 0, 1 | 0 | 0 | 1 |
| DensifyMark | 5 | 1 | 0 | 0 | 64 |
| DensifyScatter, DensifyScatterMoments | 5 | 1 | 0 | 0 | 64 |
| ResetOpacity | 4 | 0 | 0 | 0 | 64 |
| Perturb, LossDelta (grad_test only) | 3, 8 (at the limit) | 1 | 0 | 0 | 1, 256 |
| Prepare, Difference (ssim_test only) | 5, 3 | 1 | 0 | 0 | 256 |
| ZgalSortCount | 4 | 1 | 0 | 64 B (16 atomic bins) | 256 |
| ZgalSortOffsets | 1 | 1 | 0 | 260 B with subgroups, 1028 B without (256 lanes and a carry) | 256 |
| ZgalSortScatter | 6 | 1 | 0 | 8320 B with subgroups (2048 subgroup counts), 1152 B without (256 digits, 16 next, 16 bins) | 256 |
| ZgalScan* / ZgalReduce* | at most 4 (3, and the count's buffer with a GPU `#:count`) | 1 | 0 | at most 260 B with subgroups; without, 1024 B (256 lanes), 1028 B for ZgalScanApply | 256 |
| Present (vertex, fragment) | 0 | 1 | 0 (one sampled texture) | | |

The largest buffers stay within `maxStorageBufferBindingSize` (128 MiB) and so within
`maxBufferSize` (256 MiB): the rasterizer's `colours` at the programs' largest image, 2896² ×
16 B = 134,189,056 B, and the tile entries' keys at their cap of 2²⁵ entries
([architecture.md](architecture.md#render-graph)), and the training graph's `entryGradients` at
one chunk pass's most, 3·2²⁰ entries, 113,246,208 B ([architecture.md](architecture.md#train-graph)), and
its `Splat` buffers at splat_train's cap of 559240 splats (`--max-splats`), 134,217,600 B. A 1D dispatch is at most 65535 workgroups
(`maxComputeWorkgroupsPerDimension`), as natively. ZGAL enables subgroups when the adapter has
them, and the sort and scan kernels fall back to their variants without them otherwise; both fit.

## Portability

WGSL lets a store to one component of a vector in storage rewrite the whole vector, which SPIR-V
doesn't: two invocations writing different components of one element race on the web (Dawn on
Metal lost one of them), though natively they don't. So no kernel has invocations write different
components of one storage element; TileRanges' tile ranges are two `uint`s a tile rather than a
`uvec2` for this ([architecture.md](architecture.md#render-graph)). ZGAL doesn't diagnose it.

## Tests

`make test-web` runs the same tests as the native `make test-programs`, through the same
`test.sh` scripts (`test-web.sh` calls them; a `.js` binary runs under `$NODE`):

- **splat_render**: S3 cameras 0 and 24 captured and compared with the same goldens at the same
  PSNR (45 dB). It has no externs, so unlike render_test it needs no `.inputs.ts`.
- **splat_train**: the native test, its ground truth rendered by splat_render's web build and its
  re-renders too, at the same thresholds: perturb within 0.5 dB of the native builds' PSNR (the
  web's parity check: [Training on the web](#training-on-the-web)), sparse at 28 dB. Its
  `.inputs.ts` writes the PLY with Node's `fs`, so the web build's export works only under Node.
- **render_test**: every run of the native test (S1 with the oracle, the off-centre variant, S2
  twice with the hashes compared, S3). Its checks are C++ (`render_check.h` with
  `tests/oracle/closed_form.h`), so rather than port them, the web build's `check` writes the
  checked frame's readbacks to `$RENDER_TEST_READBACKS` and the host tool
  `build/web-check/render_test_check` (`programs/render_test/web-check/`) runs the same checks on
  them. A copy with corrupted colours must fail them. `make test-web` therefore needs a C++
  compiler as well as Node.
- **grad_test**: every run of the native test, in the same way: `$GRAD_TEST_READBACKS` and
  `build/web-check/grad_test_check`, and a copy with the opacity gradients raised by 1% must fail.
  The finite differences themselves run in the app, the same on both.
- **densify_test**: every run of the native test; its checks are in the app.
- **ssim_test**: every run of the native test; its checks are in the app.

## Training on the web

splat_train's web build trains as the native one does: the same graph, kernels and schedule, the
camera order and the densification's seeds from `pcgHash` (in the TypeScript host too, with
`Math.imul`), so a run differs from native only by the float rounding of the two shader
compilers. Training needs no more than WebGPU's default limits (the table above); none bit.

perturb's held-out PSNR is within 0.5 dB of native, and the test checks it; sparse's isn't in
one run, though it is on average. Measured on S3 (test.sh's two runs; `--seed` changes only the
camera order; test.sh runs seed 0):

| Run | Native | Web | Native − web |
|---|---|---|---|
| perturb, CI (lavapipe both) | 35.94 | 35.89 | 0.05 |
| perturb, M4 (MoltenVK; Dawn on Metal), `--seed` 0–4 | 35.91 to 36.49, mean 36.15 | 35.80 to 36.27, mean 36.07 | 0.08 ± 0.13, at most 0.27 |
| sparse, CI (lavapipe both) | 37.08 | 36.33 | 0.75 |
| sparse, M4, `--seed` 0–7, one ground truth | 36.53 to 37.08, mean 36.87 | 36.64 to 36.93, mean 36.80 | 0.07 ± 0.19 |

There is no systematic gap, but sparse's single run is chaotic: densification's thresholds turn
the least difference into different splats. Trained on ground truths differing in 10 of 1.6M
bytes by one level (splat_render's native and web captures of S3), the same web build scored
36.89 and 36.17 dB, native 36.99 and 37.12. So CI's 0.75 dB is noise of one run. perturb is
steadier: at test.sh's seed every device scored within 0.1 dB (35.86 to 35.94), and across
seeds the web trailed native by 0.08 ± 0.13 dB, at most 0.27.

So only perturb is held to 0.5 dB of native in the test: every build must reach test.sh's
`native_perturb_psnr` (35.9) less 0.5 dB, and a native build fails more than 1 dB above it, so
the reference is updated when training improves rather than leaving the web's bound behind (the
web isn't bounded above). sparse keeps its absolute 28 dB, the user's decision (2026-11-04):
holding it to native would need the mean of several seeds (about 9 more minutes in the web CI
job, past its step timeout) or a less chaotic run first.

## Not supported on the web

- **tile_test**: builds, but its checks are native only (its `.inputs.ts` stubs throw), so it has
  no web test.
- **The window**: splat_render's present path runs only in a page (`splat_render.zgal.html`), and
  no test covers it; the tests are headless under Node.
- **Paths in externs**: a ZGAL extern can't take a `path` (or string) parameter, so
  render_test's and grad_test's readback directories and splat_train's PLY are environment
  variables rather than options.
