# Building and testing

## Requirements

The wrapper's: the Vulkan SDK (or the system's Vulkan headers and loader), `glslc`, GLFW, GLM,
Racket 8 and, for the web, Node 24 and npm (`external/vulkan_test/docs/building-and-testing.md`
has the details and the platform paths). The data tools need Python 3 with pytest, numpy,
scipy and Pillow (`tools/requirements.txt`); a virtualenv in `.venv/` is ignored by git, e.g.
`python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt`, then
`make test PYTHON=.venv/bin/python`. Clone with `--recursive`, or run `git submodule update --init --recursive`.

## Make targets

- `make`: builds every program into `build/<name>/<name>`.
- `make test`: runs, in order,
  - `test-tools`: pytest over every `test_*.py` under `tools/` and `tests/`;
  - `test-cpp`: each `tests/cpp/<name>.cpp`, built into `build/tests/<name>` and run;
  - `test-programs`: each program's `test.sh` (`make test-<name>` runs one).
- `make web`: builds every program for the web into `build/web/<name>`.
- `make test-web`: each program's `test-web.sh` under Node (`make test-web-<name>` runs one).
- `make clean`: removes `build/`.
- `ZGAL_ROOT=<checkout>` builds against another wrapper checkout; `CXX=...` another compiler
  (clang by default); `PYTHON=...` another Python.

## Adding programs and tests

The Makefile loops over what's on disk, so adding these needs no Makefile change:

- **A program**: `programs/<name>/<name>.zgal` with a `main`, built with `zgal.mk`'s
  `zgal-program` (its `.cpp` files beside it, `assets/` copied beside the binary) and
  `zgal-web-program` (its `.ts` files). Every program is built for both, so it must also build
  for the web (no native-only features such as `import-images`). Programs link only Vulkan; one
  that needs more, e.g. GLFW for a window, opts in with `programs/<name>/program.mk`:
  `LIBS_<name> = -lglfw`. Headless programs then build and run without GLFW.
- **A program test**: `programs/<name>/test.sh`, executable, run from the repository's root as
  `test.sh <binary> <output dir>`. It runs the program headless with `--validation` (which fails
  unless validation and synchronization validation are both available) and exits non-zero on any
  failure; a validation message is a failure.
- **A web test**: `programs/<name>/test-web.sh`, run as `test-web.sh <main.js> <output dir>`
  (`node <main.js> ...` runs it under Node with Dawn).
- **A C++ test**: `tests/cpp/<name>.cpp`, a whole program (doctest with
  `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`, or its own `main`) with `tests/`, doctest and the wrapper's
  headers on the include path, and no library linked, so it tests header-only code (the oracles).
  It's rebuilt when any header in `tests/cpp/` or `tests/oracle/` changes.
- **A Python test**: `test_*.py` anywhere under `tools/` or `tests/`. They run in one pytest
  invocation without `__init__.py` files, so each basename must be unique (pytest otherwise fails
  with "import file mismatch").

## CI

GitHub Actions (`.github/workflows/`, cloned from ppu's) checks every commit a push brings: on a
working branch, each commit beyond `main` (the newest 50); on `main`, the pushed range.

- `ci.yml`, on every push:
  - `commits` lists the commits (the wrapper's `commits` action, from the pinned submodule);
  - `pending` marks each commit's `test` status pending;
  - `test` runs `test.yml` for them (the newest 50 beyond main);
  - `statuses` posts each commit's `test` status from its job's outcome.
- `test.yml` runs one job per commit (`test <sha>`) on ubuntu-24.04, set up by the wrapper's
  `setup-zgal` action (native toolchain: lavapipe, the Vulkan SDK with its validation layers,
  clang through ccache, Racket and Node), plus pytest, numpy, scipy and Pillow from Ubuntu's archive (7.4, 1.26, 1.11 and 10.2 on 24.04). It
  runs `make all web`, then `test-tools`, `test-cpp`, `test-programs` (native, on lavapipe, with
  validation) and `test-web` (Node with Dawn on lavapipe), each running even if an earlier test
  failed, and with `make -k`, so one failing program doesn't stop the others. On failure it uploads `build/test` and `build/test-web`.
- Permissions: the jobs that run code (the repository's, the wrapper's or third-party actions)
  have a read-only `contents: read` token and check out without persisting credentials; only
  `pending` and `statuses`, which check nothing out, can write commit statuses.
- lavapipe's subgroup size is 4, 8 or 16 depending on the runner's CPU, so a kernel that assumes
  one may pass locally and fail here, or the other way round.

## Closed-form oracles

`tests/oracle/closed_form.h` holds hand-derived single-splat renders, the two-splat composite and
single-splat L2 gradients for one canonical setup (camera at the origin looking down +z, one focal
length, splats rotated only about z, DC colour), in double; the header documents each formula's
derivation and paper equation. Kernel tests compare against it; `tests/cpp/closed_form_test.cpp`
tests it on its own. When using it:

- `render()` has no tile cutoff: compare whole images only for a splat whose 3σ square covers the
  image (`footprint_covers_image()`), or mask the tiles the splat doesn't touch.
- The gradients are defined only where no pixel's alpha is clamped at 0.99 (they throw otherwise);
  pixels below 1/255 contribute nothing.
- Feed kernels the rotation as `quat_about_z(theta)`. For a camera rotated about z, feed the world
  splat and compare with the oracle on `world_to_camera_about_z()`'s splat; that guards the
  covariance's W against Wᵀ.
- `projected_l2()` gives the gradients of the projected stage (2D mean, Σ′ or the conic, activated
  opacity, colour) for the backward rasterize; `single_splat_l2()` chains them to the splat's
  parameters. A symmetric matrix's `b` gradient is for the shared off-diagonal value (both entries).
  Against the reference: its conic gradient's off-diagonal is half of `conic.b`, its Σ′ gradient
  equals `cov.b`, and its `dL_dmean2D` is per NDC unit (0.5·W, or 0.5·H for y, times `mean`).

## Synthetic scenes

`tools/make_synthetic.py` writes test scenes in the converter's format (`docs/data-format.md`),
with cameras and splats but no images; its tests are `tools/tests/test_synthetic.py`. Each scene
is a pure function of its arguments (the seed defaults to 0), so the same command writes the same
bytes with the same numpy version on the same platform, and the manifest's `source` records the
generator and its arguments. Across numpy versions or machines the bytes may differ: numpy doesn't
promise stable random streams, and float64 maths and BLAS can differ in the last bit. So a golden
(a capture compared across runs) must commit the scene it was made from, not regenerate it in CI.

```
tools/make_synthetic.py s1 <out dir> [--variant <name>]
tools/make_synthetic.py s2 <out dir> [--seed <n>]
tools/make_synthetic.py s3 <out dir> [--seed <n>] [--count <n>]
tools/make_synthetic.py perturb <scene dir> <out dir> [--seed <n>] [--jitter <f>]
tools/make_synthetic.py sparse <scene dir> <out dir> [--seed <n>] [--fraction <f>]
```

- **S1** (`<out dir>/<variant>/`, no seed): the closed-form oracles' setups, with
  `tests/cpp/closed_form_test.cpp`'s numbers. Cameras sit at the origin, rotated only about z,
  with fx = fy = f and the principal point at the image centre; the splats have DC colour only and
  rotation `quat_about_z(theta)`. `source.oracle` holds what a kernel test needs to build the
  oracle's `Camera` and `Splat`s: `f`, `width`, `height`, `background` (black where the C++ test
  sets none), `target` (the gradient setup only), `cameraPhi` (camera k is Rz(phi_k); for phi ≠ 0
  compare with `world_to_camera_about_z(splat, phi)`) and the world-space `splats` (`x`, `y`, `z`,
  `logScale`, `theta`, `opacityLogit`, `dc`).

  | Variant | Setup (f, size) | What it pins down |
  |---|---|---|
  | `isotropic` | 50, 33² | isotropic on the axis, theta 0.3: a = c = (f s / z)² + 0.3, b = 0 |
  | `anisotropic`, `-rot90`, `-rot45` | 50, 33² | axis-aligned; axes swapped; b ≠ 0 with the same eigenvalues |
  | `off-axis` | 50, 33² | the Jacobian's off-axis terms (sz matters) |
  | `one-splat`, `-clamped` | 50, 33² | the per-pixel value over a background; alpha clamped at 0.99 (logit 12) |
  | `gradient` | 20, 15² | `single_splat_l2()`: footprint covers the image, nothing clamped |
  | `two-splat` | 50, 33² | the composite in depth order; stored back splat first, so the kernel must sort |
  | `rotated-camera`, `-aligned` | 50, 33², 3 cameras | W against Wᵀ, with theta 0.7 and 0 |

  The 33-pixel images put a pixel centre on the principal point. The cameras all sit at the
  origin, so the converter's extent (from camera centres) would be 0; S1 records instead 1.1 times
  the largest distance from the origin to a splat, so a training run on S1 gets a non-zero
  position learning rate. Only `gradient`'s footprint
  covers the image, so compare the others whole-image only where `footprint_covers_image()` says
  so, or mask by tile.
- **S2** (seeded): 8 splats, for exact tile ranges and finite differences. Five overlap near the
  axis at depths 4.00–4.08 (0.5% apart, which the depth key must still order), one has its mean
  just past the right edge with its footprint reaching in, one is entirely off-screen and one is
  behind both cameras (`source.cases` names each, as `cam_0` sees it). Positions are fixed; the
  anisotropic scales, unnormalized random rotations, opacities and all SH bands come from the
  seed. `cam_0` is 64×64 at the origin (R = I); `cam_1` is 80×48, off the axis, looking at the
  cluster.
- **S3** (seeded): `--count` (2000) Gaussians uniform in the unit ball, with anisotropic scales of
  0.01–0.06, unnormalized random rotations, opacities 0.3–0.95 and SH degree 3, seen by 24 128×128
  cameras on a Fibonacci sphere of radius 4 looking at the origin (world +z up) and one 160×96
  camera (`cam_24_wide`), all with f = 140 so every view holds the whole ball. Every 8th sphere
  camera is held out (0, 8, 16); the 160×96 camera is always a training view, so re-fitting runs
  the backward pass on a non-square image.
- **perturb** and **sparse** copy a scene (images and all, so run them after its images are
  rendered) into a new or empty directory outside it, with degraded splats as a training start.
  perturb jitters positions by Gaussian noise of σ = `--jitter` (0.05) × the positions' RMS distance
  from their centroid, doubles or halves each splat's scale on all axes, draws DC colours in [0, 1]
  (higher bands zero) and opacities in [0.05, 0.95], and keeps rotations. sparse keeps a random
  `--fraction` (0.1, at least 4) of the positions, in order, and initializes splats from them as the
  converter does from COLMAP points, coloured with the original DC colours clamped to [0, 1].

## GPU runs

Run GPU work (program tests, captures) one at a time on a machine. Locally that's MoltenVK on the
Apple M4; a GPU test is done only when it passes with validation and synchronization validation and
prints no validation message.

## Bumping the wrapper

`external/vulkan_test` is pinned to a commit of github.com/zach-youssef/vulkan_testing. Bump it to
a pushed commit (`git -C external/vulkan_test checkout <sha>`, then commit the submodule), and
read that repository's `docs/consumers.md` and `docs/MIGRATING.md` for what changed.
