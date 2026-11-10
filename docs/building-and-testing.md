# Building and testing

## Requirements

The wrapper's: the Vulkan SDK (or the system's Vulkan headers and loader), `glslc`, GLFW, GLM,
Racket 8 and, for the web, Node 24 and npm (`external/vulkan_test/docs/building-and-testing.md`
has the details and the platform paths). The data tools need Python 3 with pytest, numpy,
scipy and Pillow (`tools/requirements.txt`); a virtualenv in `.venv/` is ignored by git, e.g.
`python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt`, then
`make test PYTHON=.venv/bin/python`. Clone with `--recursive`, or run `git submodule update --init --recursive`.
The data tools also call C++ through the native library ([below](#the-native-library)), so they
need it built (`make native`, which the test targets do first), and with it the C++ compiler and
Racket (for its ZGAL-generated header).

## Make targets

- `make`: builds every program into `build/<name>/<name>`, and the native library
  ([below](#the-native-library)).
- `make native`: builds the native library alone.
- `make test`: runs, in order,
  - `test-tools`: pytest over every `test_*.py` under `tools/` and `tests/`, after building the
    native library;
  - `test-cpp`: each `tests/cpp/<name>.cpp`, built into `build/tests/<name>` and run;
  - `test-programs`: each program's `test.sh` (`make test-<name>` runs one), after building the
    native library (their checks run the data tools).
- `make web`: builds every program for the web into `build/web/<name>`.
- `make test-web`: each program's `test-web.sh` under Node (`make test-web-<name>` runs one),
  after building its web-check tools and the native library; it needs Python as `make test` does
  (`PYTHON=...`).
- `make test-pages`: each program's `test-page.sh` (`make test-pages-<name>` runs one), after its
  web build and the native library (a page test may encode its scene): its page in headless
  Chromium on SwiftShader, through the wrapper's `web/tools/chromium.ts`. It needs Python as `make test` does, and Chromium's headless shell:
  `CHROMIUM=<path>`, else the newest one Playwright installed
  (`npx playwright install chromium-headless-shell`). A missing Chromium fails it, unlike the
  wrapper's `make test-pages`, which skips one it doesn't find.
- `make clean`: removes `build/`.
- `ZGAL_ROOT=<checkout>` builds against another wrapper checkout; `CXX=...` another compiler
  (clang by default); `PYTHON=...` another Python.

## The native library

Logic that both the data tools (Python) and the C++ programs need lives once, in C++: header-only in
`lib/native/` ([architecture.md](architecture.md)). The tools call it through
`build/native/libsplat_native.so` (`.dylib` on macOS), which make builds from
`lib/native/c_api.cpp`, a C ABI over those headers (plain structs and status codes, nothing thrown
across it), with `-O2`; `tools/splatdata/native.py` binds it with ctypes (the standard library, so
nothing is vendored) and loads it on first use: make's build, whose path make exports to its recipes
as `$SPLAT_NATIVE_LIB` (so `OUT=...` builds and tests its own), or `build/native/`'s outside make.
ctypes doesn't check the signatures, so the binding refuses a library whose structs' sizes differ
from its own, and `tools/tests/test_native.py` calls every C function. Adding a function means: the
header in `lib/native/` (and its C++ test), its wrapper in `c_api.cpp`, its signature in
`native.py`, and a test in `test_native.py`. The library's generated header comes from
`lib/native/c_api.zgal`, which imports the modules whose structs it uses.

It holds the INRIA PLY reader (`lib/native/ply_reader.h`), which `splatdata.splats.read_ply`
calls, so the data tools' tests and every tool reading a PLY need the library built, and the
splats.bin reader (`lib/native/splats_bin_reader.h`). Both return a `ReadStatus`
(`lib/native/read_status.h`), which the binding raises as `OSError`, `ValueError` or `MemoryError`. It
also holds the stream encoder and the default camera (`lib/native/stream_encoder.h`,
`default_camera.h`; [data-format.md](data-format.md#stream-files)): the binding mirrors the
encoder's `EncodeConfig` and `EncodeReport` as ctypes structs and raises a refusal as
`native.EncodeError`, naming the group and splat. The PLY writer is the one piece still
duplicated in Python ([data-format.md](data-format.md#inria-ply)). Programs include the headers
directly, without the library: splat_view's native opener reads and encodes a PLY with them
(`programs/splat_view/program.mk` makes its object depend on `lib/native/*.h`).

## Adding programs and tests

The Makefile loops over what's on disk, so adding these needs no Makefile change:

- **A program**: `programs/<name>/<name>.zgal` with a `main`, built with `zgal.mk`'s
  `zgal-program` (its `.cpp` files beside it, `assets/` copied beside the binary) and
  `zgal-web-program` (its `.ts` files). Every program is built for both, so it must also build
  for the web (no native-only features such as `import-images`). Programs link only Vulkan; one
  that needs more, e.g. GLFW for a window, opts in with `programs/<name>/program.mk`:
  `LIBS_<name> = -lglfw`. Headless programs then build and run without GLFW.
  A ZGAL program binary loads its shaders from `shaders/` beside itself, so a copy of one (e.g. a
  mutant's binary kept beside the real build) needs that directory copied with it.
  A numeric option whose every value, 0 included, is a meaningful choice is made required with
  ZGAL's `#:given`: the option also sets a `bool` parameter when it's on the command line, and the
  program fails without it (splat_train's `--max-screen-size`, where 0 turns the prune off).
- **A program test**: `programs/<name>/test.sh`, executable, run from the repository's root as
  `test.sh <binary> <output dir> <web-check dir>`, after the program's web-check tools (a test
  that needs none ignores the third argument). It runs the program headless with `--validation` (which fails
  unless validation and synchronization validation are both available) and exits non-zero on any
  failure; a validation message is a failure.
  A POSIX sh function's variables are global, so a helper that assigns one (splat_view's
  `view()` sets `name`) overwrites a caller's variable of that name: a loop around such a call
  needs another name (splat_view's `case_name`). For a per-run environment variable around a
  function call, use a subshell with `export`: an assignment before a function call may outlive it.
  A training test ends several hundred iterations after its last densification or opacity reset
  (splat_train's `--densify-until 1500` in 2000 and 1800-iteration runs): right after one, fresh
  clones and splits are untrained and a reset has just cut every opacity, so the score says
  little about the trainer (a sparse run ending just after a reset scores about 13 dB instead of 33).
  The last iteration itself never densifies or resets (paper-conformance.md, departure 17),
  which splat_train's `last` run checks.
  A test that runs another program (splat_train's renders its ground truth with splat_render)
  names that build in its `program.mk`: `test-<name>: $(OUT)/<other>/<other>` and
  `test-web-<name>: $(OUT)/web/<other>`.
- **A web test**: `programs/<name>/test-web.sh`, run from the repository's root as
  `test-web.sh <main.js> <output dir> <web-check dir>` (`node <main.js> ...` runs the program
  under Node with Dawn, where WebGPU validation errors fail a `--validation` run). The programs'
  `test-web.sh` call their `test.sh`, which runs a `.js` binary under `$NODE` (`node` by default),
  so the web runs the native tests ([web.md](web.md#tests)).
- **A page test**: `programs/<name>/test-page.sh`, run from the repository's root as
  `test-page.sh <web build dir> <output dir> <chromium.ts>`. It serves the program's page
  (`<name>.zgal.html`, its options and positional arguments in the query string) and its inputs
  from one directory under its output directory (`chromium.ts --serve`, which follows symlinks;
  `tools/scene_args.py --query <url dir> ...` prints a scene's arguments as that query string),
  runs it in Chromium, and checks what it shows (`--screenshot`, `--keys`), failing when the
  checker exits non-zero, 3 (no Chromium) included. SwiftShader is slow, so a page test uses a
  tiny scene: splat_render's and splat_view's render S3 (2000 splats) at 128x128
  ([web.md](web.md#tests)).
- **A web-check tool**: `programs/<name>/web-check/<tool>.cpp`, a host program a test runs on
  what a run wrote, for checks written in C++: a web run's (render_test's `render_test_check`), or
  either backend's (splat_view's `view_oracle_check`). It's built
  like a C++ test (with `tests/` on the include path, and a `<tool>.zgal` beside it for a generated
  header) into `build/web-check/<tool>` before the program's `test-<name>` and `test-web-<name>`, and rebuilt when
  the program's or `tests/oracle/`'s headers change. Tool names must be unique across programs.
- **A C++ test**: `tests/cpp/<name>.cpp`, a whole program (doctest with
  `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`, or its own `main`) with `tests/`, doctest and the wrapper's
  headers on the include path, and no library linked, so it tests header-only code (the oracles).
  It's rebuilt when any header in `tests/cpp/`, `tests/oracle/` or `lib/native/` changes. To test a ZGAL module's
  functions, add `tests/cpp/<name>.zgal` beside it (importing the module): the test then includes
  the generated `<name>.zgal.h` (namespace `<name>`), and the file's shaders are compiled too, so a
  compute material there that calls the functions has glslc check them
  (`tests/cpp/splat_math_test.zgal`).
  The generated C++ is compiled like any C++, so clang may contract `a + b · c` into an FMA (one
  rounding instead of two): a test comparing it bit for bit with its own float32 arithmetic accepts
  either result (`low + stepSize · q` in `splat_stream_test.cpp`).
  Seeded random test data comes from `tests/cpp/test_random.h` (`TestRandom`), not the `<random>`
  distributions: `std::mt19937`'s words are standard, but `uniform_real_distribution`,
  `normal_distribution` and `uniform_int_distribution` aren't, and libstdc++ (CI's Linux) and
  libc++ (macOS) draw different numbers from the same seed, so a test would check other cases on
  each and could pass on one only. `make test-cpp CXX=g++-16 OUT=build/gcc` (Homebrew's GCC, with
  libstdc++) runs the C++ tests against Linux CI's standard library (CI compiles with clang++).
  Keep the draws in a defined order: several in one value through brace initialization
  (`glm::vec3{u(rng), u(rng), u(rng)}`, evaluated left to right), never as the arguments of one
  call or parenthesized constructor, whose order is unspecified (GCC on x86-64 can go right to
  left).
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
- `test.yml` runs one job per commit and suite (`test <sha> <suite>`) on ubuntu-24.04, set up by
  the wrapper's `setup-zgal` action (native toolchain: lavapipe, the Vulkan SDK with its
  validation layers, clang through ccache, Racket and Node). The suites are the entries of its
  matrix's `suite` list, each with the make targets it builds and runs, so a new shard is one
  entry:
  - `native`: `make all`, then `test-tools` (pytest, with numpy, scipy and Pillow from Ubuntu's
    archive: 7.4, 1.26, 1.11 and 10.2 on 24.04), `test-cpp` and `test-programs` (on lavapipe,
    with validation). The data tools' and the C++ tests need no program, so only this shard runs
    them (`cpu-tests`); every other shard would repeat them.
  - `web-splat_train` and `web-rest`: `make web`, then the web tests (Node with Dawn on
    lavapipe), which build the native library first (splat_train's check reads PLYs through
    it), split in two by time: `test-web-splat_train`, and `test-web TEST_WEB_SKIP=splat_train`
    (`TEST_WEB_SKIP` lists programs `test-web` leaves out). `web-rest` runs every program not
    skipped, so a new program's web test runs without being assigned to a shard; to move one to
    the first shard, add it to both lists.
  - `pages`: `make web`, then `test-pages` (headless Chromium on SwiftShader), which builds the
    native library first (splat_view's page test encodes its stream). Only this shard
    (`chromium`) installs Chromium's headless shell, with the steps of vulkan_test's `web.yml`:
    Playwright's (`PLAYWRIGHT_VERSION`, cached), after lifting Ubuntu 24.04's AppArmor ban on the
    unprivileged user namespaces its sandbox needs, and checked to be exactly one shell, which
    `CHROMIUM` then names. Its page tests take well under a minute (on an M4, splat_render's
    page about 9 s, splat_view's two about 14 s with the encoding), so the job is mostly setup,
    the web build and the install.
  Each test step runs even if an earlier one failed, and the program tests with `make -k`, so one
  failing program doesn't stop the others. On failure or cancellation a job uploads `build/test`,
  `build/test-web` and `build/test-pages` (each program's logs, as `test-outputs-<sha>-<suite>`);
  the program tests' step has a 20-min timeout of its own, so it fails before the job's 30 and
  the upload still runs.
  `ci.yml` posts a commit's status from all its jobs: success only if every suite succeeded.
- Why shards: in one job the program tests ran one after the other and took 17.7 min of a
  24.2-min job (run 41460382408), and D-SSIM adds per-step cost to every training test. Running
  native and web together with `make -j2` was no faster (the program-test step took 15 to 21 min
  across runs, within the runners' +-25% variance), since lavapipe spreads its rasterizing over
  every vCPU of the 4-vCPU runner, so the two tests contend. Separate jobs run on separate
  runners. The web suite in one job took 11.5 to 20 min and timed out (runs 42717265046 and
  42721091082, where the same commit's times varied by up to 1.7x between runners), so it is
  split: splat_train's web test alone takes 6.5 to 12 min, every other one's 5 to 8 (grad_test
  4 to 6.5 of them). The native suite's take 8.5 to 14 min, the slowest shard, against the
  20-min step timeout, which is what the timeouts are sized from. A stack of N commits is 4N
  jobs, against GitHub's 20 concurrent. A program's tests must not share files
  with another program's or with the other suites' (each writes only to its output directory,
  `build/test/<name>`, `build/test-web/<name>` or `build/test-pages/<name>`). Watch the slowest
  job's time when adding a test.
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
- `projected_gradients()` gives the gradients of the projected stage (2D mean, Σ′ or the conic,
  activated opacity, colour) for the backward rasterize, with the L2 loss (Σ(I − t)²,
  `projected_l2()`) or the L1 (the mean |I − t| over 3WH, training's L1 term); `single_splat_l2()`
  chains the L2's to the splat's parameters. A symmetric matrix's `b` gradient is for the shared off-diagonal value (both entries).
  Against the reference: its conic gradient's off-diagonal is half of `conic.b`, its Σ′ gradient
  equals `cov.b`, and its `dL_dmean2D` is per NDC unit (0.5·W, or 0.5·H for y, times `mean`).
- A program's check reads its scene through `tests/oracle/zgal_scene.h`: the oracle's camera and
  splats from `lib/splat_math.zgal`'s `Camera` and `Splat` (`sceneCamera()`, `sceneSplat()`, which
  refuse anything but S1's setup), the borderline rule for a pixel whose alpha or T sits within
  rounding of its threshold (`Borderline`, `borderline()`) and the hash the checks print.

## Gradient checks

`programs/grad_test` checks `TrainGraph`'s gradients
([architecture.md](architecture.md#train-graph)) against the closed form and the CPU, and by finite
differences:

- **Closed form**: S1's `gradient` variant against `projected_gradients()`, L2 and L1, within 1e-3
  relative (or 1e-2 of the largest gradient of its kind, for one near 0), and the parameters'
  against `single_splat_gradients()` (the in-plane position, the log-scales, the opacity logit
  and the SH DC; L2 and L1). The targets are 8-bit, so the oracle takes the target rounded as the
  file has it.
- **On every run**: the parameter gradients (`parameterGradients`) must match the CPU's
  `preprocessBackward` (the C++ ZGAL generates from `lib/splat_backward.zgal`) on the read-back
  projection gradients within 1e-3 relative (to the larger of the value and 1e-2 of the largest of
  its group: position, opacity, log-scale, quaternion, SH), and the densification's statistics
  must be consistent: every visible splat seen the same n frames, culled ones none, and each norm
  n times `densifyNorm` of its 2D-mean gradient within 1e-4, and its largest radius
  (`maxRadius`) the frames' `Projected` radius (0 for a culled one). `tests/cpp/splat_backward_test.cpp` checks
  `preprocessBackward` itself on the CPU, against the oracle and central differences of
  `projectSplat` (general rotations and cameras, SH degrees 1 to 3, the clamps).
- **GPU finite differences**, the plan's rule: the smooth L2 loss against a fixed target unlike
  the render (`scenes.py` writes a sin/cos pattern). The probes are the visible splats' opacity,
  colour and 2D mean, or with `--parameters` their parameters (the position, the opacity logit,
  the log-scale, the quaternion, and each SH band up to the scene's degree, the DC on its own),
  chosen without looking at the analytic gradients, so a gradient wrongly 0 or small is probed
  too: of each kind, every field, or `--samples` (default 32) spread evenly over them on a bigger
  scene. Central differences with h = 1e-3 of the value (at least 1e-4) for the opacity and
  colour, and a fixed 0.005 px for the mean (a step relative to a pixel position means nothing,
  and 0.05 px left a few % of truncation error on S3's smaller splats); for the parameters 1e-3
  of the value (at least 1e-3), and for the position 0.005·depth/fx, which moves the 2D mean
  about 0.005 px. An SH coefficient whose colour channel is within 0.01 of its clamp at 0 isn't
  probed (the step would cross the kink). A probe passes when the analytic gradient and the FD
  are both finite and |analytic − FD| ≤ 2e-2·max(|analytic|, |FD|) + 1e-5 (for the quaternion,
  + 2e-3·largest), or both are negligible. Each kind needs at least
  one probe that isn't negligible. The perturbation is to the projection, or with `--parameters`
  to the splat, projected again (keeping its depth and tiles, so the sort and the entries stay
  the same), one frame per side, and the loss's change between the sides is summed pixel by pixel
  by a `reduce`.
  - Negligible: max(|analytic|, |FD|) below 1e-2 of the largest analytic gradient of its kind
    (opacity, colour, mean, or with `--parameters` each parameter kind above) among the visible
    splats. Summing the per-pixel changes in float leaves an absolute error of about 1e-3 of the
    largest gradient, which fails the rule's relative bound on smaller gradients. The
    quaternion's mid-sized gradients sit in that noise (S2's 0.47 was off by 0.03 beside the
    kind's largest, 21), hence its 2e-3·largest; a larger step trades the noise for truncation
    error and for more flipped, excluded pixels (at 1e-2 of |q| the FD was off by 0.05: the
    rotation of an anisotropic splat isn't close to linear over it). The other kinds keep the
    plain rule.
  - The loss jumps where a splat's α crosses 1/255 (a jump adds up to about 4 to a difference) and
    where the stop at T < 1e-4 moves to another entry, so pixels where either flips between the
    two sides are left out ([architecture.md](architecture.md#train-graph)). Without the first, 12
    of 36 differences failed on S2.
  - Where the oracle can't reach, test.sh runs: S2's two cameras; S1's `one-splat-clamped` (α
    clamped at 0.99, whose derivative is exactly 0); S1's `stack`, with `--stops` (fails unless a
    pixel stops at T < 1e-4, counted on the CPU from the projections); S3's cameras 0 and 24 with
    `--samples 24` and `--multi-batch` (fails unless a tile's backward walks more than one batch,
    from `backwardRanges`). With `--parameters`: S2's two cameras (SH degree 3, so every band),
    and S3's camera 0 with `--samples 24` (with 8, every probe of the third band was negligible).
    With `--l1 --dssim 0.2` (training's 0.8·L1 + 0.2·D-SSIM): S2's two cameras.
  - With `--dssim`, only the colour is probed, with h = 1e-2 of the value (at least 1e-3). A
    pixel's D-SSIM depends on its 11×11 window, so leaving out a flipped pixel doesn't remove its
    colour's jump from its neighbours' losses: opacity and mean steps were up to 15% off on S2
    (one of opposite sign), and leaving out the window around each flip dropped too much of the
    smooth change instead. A colour step flips nothing, and the opacity and mean share the
    backward pass from the per-pixel gradient on, which the L1 and L2 runs check. At 1e-3 the
    SSIM's float noise (its variances are differences of near-equal sums) left colours a few %
    off; at 1e-2 they agree within 1.1%. `--dssim` refuses `--parameters`, `--oracle` and the L2 loss.
- `--target-pixels` (the pixels the targets file holds; `scenes.py` passes it) is required in
  effect: a camera whose pixels at `--target-offset` don't fit fails up front, which test.sh
  checks with a too-small value.
- Every run checks that the gradients are finite and prints a hash; test.sh compares two runs'
  hashes. The runs without `--finite-differences` also check that the next `--repeats` frames
  give the same bits.
- **Chunks**: `--chunk-entries` sets how many tile entries one backward pass holds
  ([architecture.md](architecture.md#train-graph)). test.sh checks that several passes give the
  same hash as one: S2's camera 0 (26 entries) at 5 entries a pass, and S3's camera 0 (5269
  entries) at 97, so a pass boundary falls inside a tile and inside a splat's entries. More
  passes than the graph holds (`trainChunkPassesMax`, 64) must fail up front: `--capacity 6500
  --chunk-entries 100`.

## Densification check

`programs/densify_test` checks one densification frame ([architecture.md](architecture.md#train-graph))
on the GPU, every check in the app (so the web runs them too). On S3 with every 16th splat faded
(opacity logit −7), from camera 0, it trains 20 Adam steps (L2 against `scenes.py`'s target),
idles until the reads show an idle frame (from the frames in flight + 3 after the phase starts), reads
the splats, moments, statistics and counts, runs the densification (thresholds from those
statistics: the gradient's mean, the mean largest scale of the splats reaching it, and with
`--screen-prune` 1.25 times the mean largest scale), idles again and compares:

- each splat's action with `densifyAction` on the CPU (except within 1e-5 relative of a boundary),
  and the slots with the exclusive scan of the actions' flags;
- every kept or cloned splat and clone copy bit for bit at its slot, with its moments (a copy's
  zero), and a split's halves against `splitPosition` (within 1e-3 of the largest scale) and
  `splitLogScale` (1e-5), the rest bit for bit and their moments zero; a split whose second half
  is past the capacity unchanged;
- the counts (in use, overflow, densifications, pruned, added) and the cleared statistics.

`test.sh` runs it three ways: `--max-splats 8192`; at the splat count with `--overflow` (some
additions dropped, at least one of them a split's, which must stay undone); and with
`--screen-prune --max-screen-size 5 --reset` (more opaque splats pruned than by their radius alone,
so a size prune fires, and at least one by its radius alone; the reset leaves every opacity logit
at most logit(0.01) with its moments zero). Each must see every action.

## SSIM check

`programs/ssim_test` runs `lib/ssim.zgal`'s four kernels alone (λ = 1) on an image pair, every
check in the app (so the web runs them too). `make_pair.py <case> <out dir> [--width W]
[--height H]` writes a pair (32×32 by default; ssim_test requires at least 11×11, the window; the
render as vec4 floats, the target RGBA8; numpy's `default_rng`, so the pair doesn't depend on the
C++ library) and prints ssim_test's arguments, with `--expected-ssim` the pair's SSIM in
float64 numpy (`tools/splatdata/metrics.py`, the reference's definition: zero padding, C1, C2). The app checks:

- the SSIM (1 − the reduced loss) within `--tolerance` (1e-5) of `--expected-ssim`, which is
  required (a `#:given` option);
- with `--interior-ssim`, every pixel whose window lies inside the image has S within the
  tolerance of it (the constant pair's closed-form luminance term);
- with `--zero-gradients`, every |dL/dx| ≤ 1e-7 (an image against itself);
- with `--finite-differences`, `--probes` (32) central differences at `--step` h (0.01) and 2h,
  Richardson-extrapolated ((4·D(h) − D(2h)) / 3, so the h² error cancels), against the analytic
  gradient within 1e-2 relative plus 1e-3 of the largest |dL/dx|. The probes are the four corners
  (the padding's edge), then `pcgHash(seed + k)` pixels and channels;
- every value finite.

`test.sh` runs the three cases with validation: `identity --zero-gradients`, `constant` (at
`--tolerance 2e-4`: a constant image's variance G∗x² − μ² cancels in float to about 1e-7·x²,
which against C2 = 9e-4 moves S by up to about 1e-4, as in the reference's float32 SSIM) and
`random --finite-differences` at 40×24 (non-square, so a kernel swapping the width and height
fails).

## Stream decode check

`programs/stream_test` runs `lib/splat_stream.zgal`'s decode kernels
([data-format.md](data-format.md#decode)) on a stream file, read through `buffer-file`, every
check in the app (so the web runs them too, `checkStreamHeader` through its TypeScript export).
Each frame it makes `--chunk` more bytes of the stream arrive (`arrived` = min(`--stream-bytes`,
frame × chunk)): a small kernel (CopyArrived) copies only the arrived words into the buffer the
decode reads and fills the rest with 0xFFFFFFFF (a NaN as f32 or f16, all ones quantized), and
after the decode another (CheckUndecoded, `lib/stream_check.zgal`) counts, into `counts[7]`, the splats past the decoded
records that aren't still zero. So a kernel that decodes a record before it has arrived fails,
even though a later frame decodes it again correctly. It checks each counts readback on its own
(PlanDecode records its `arrived`):
before the header nothing decoded (and, for a chunk under 432, at least one such readback: a
read event handles only the newest finished copy, so with a larger chunk frame 0's may never be
handled); after it, the code is `checkStreamHeader`'s (or 11 above
`--splats`, the working buffer's capacity) and equals `--expect-code`, and the decoded counts are
exactly the complete records, never going back, with no splat written past them. A stream that passes must be seen partly decoded when some
frame's `arrived` holds the header but not the whole stream. Once every byte has arrived, every one of a
splat's 59 values must match `--reference` within `--tolerances` (two N × 60 float32 files in the
`Splat` layout): a tolerance of 0 means bit for bit; and the header PlanDecode copied out (its
camera included) must match the stream's first 108 words. `--splats`, `--stream-bytes`, `--chunk` and
`--expect-code` are required (`#:given`); the chunk and the size must be multiples of 4.

`make_case.py <dir> [--refused] [--set FIELD=VALUE ...]` writes the reference from the format's
definition in numpy, independently of the decoder: f32 values as they are, f16 through numpy's
float16 (both at tolerance 0), and quantized ones as low + step·q (q read from the stream, so a
rounding tie doesn't matter) within 4 ulp of max(|low|, |high|). `--set` patches the header
(`fields.N.key` for a field), and `--refused` writes zeros, since nothing may decode. It prints
stream_test's arguments.

`test.sh` writes fixtures with `tools/tests/stream_fixture.py` and `convert.py stream`, and runs
with validation: the presets full, f16 and web and the configs in `configs/` (u16 at degree 1, u8
with per-component ranges at degree 2, f16 and mixed at degree 0) at 300 splats in 1000-byte
chunks (a record split across chunks); full all at once; web at 20 splats one word a frame; and
headers refused for each `checkStreamHeader` code (1 to 10) and for a valid header at the cap (559,240
splats) above `--splats` (11). Two runs must fail on a check's `MISMATCH` (not a crash, a
validation message or the frames running out): the f16 case expecting code 1, and with one
reference value's lowest bit flipped.

## Synthetic scenes

`tools/make_synthetic.py` writes test scenes in the converter's format (`docs/data-format.md`),
with cameras and splats but no images; its tests are `tools/tests/test_synthetic.py`. Each scene
is a pure function of its arguments (the seed defaults to 0), so the same command writes the same
bytes with the same numpy version on the same platform, and the manifest's `source` records the
generator and its arguments. Across numpy versions or machines the bytes may differ: numpy doesn't
promise stable random streams, and float64 maths and BLAS can differ in the last bit. So a golden
(a capture compared across runs) must commit the scene it was made from, not regenerate it in CI.
splat_render's does: `programs/splat_render/golden/s3` (S3, seed 0) with its captures.

```
tools/make_synthetic.py s1 <out dir> [--variant <name>]
tools/make_synthetic.py s2 <out dir> [--seed <n>]
tools/make_synthetic.py s3 <out dir> [--seed <n>] [--count <n>]
tools/make_synthetic.py perturb <scene dir> <out dir> [--seed <n>] [--jitter <f>]
tools/make_synthetic.py sparse <scene dir> <out dir> [--seed <n>] [--fraction <f>]
tools/make_synthetic.py render-gt <scene dir> <renderer> [--background <g>] [--frames <n>] [--validation]
                                  [--sh-degree <d>]
```

- **S1** (`<out dir>/<variant>/`, no seed): the closed-form oracles' setups, with
  `tests/cpp/closed_form_test.cpp`'s numbers (and `off-centre`, `stack`, `faint` and `faint-overlap`, which no C++ test has). Cameras
  sit at the origin, rotated only about z, with fx = fy = f and the principal point at the image
  centre (offset by (`dcx`, `dcy`) for `off-centre` and `faint-overlap`); the splats have DC colour only and
  rotation `quat_about_z(theta)`. `source.oracle` holds what a kernel test needs to build the
  oracle's `Camera` and `Splat`s: `f`, `width`, `height`, `dcx` and `dcy` (the principal point's
  offset), `background` (black where the C++ test sets none), `target` (the gradient setup only),
  `cameraPhi` (camera k is Rz(phi_k); for phi ≠ 0 compare with `world_to_camera_about_z(splat, phi)`) and the world-space `splats` (`x`, `y`, `z`,
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
  | `off-centre` | 50, 48×32, (cx, cy) = (21, 12.5) | non-square and an off-centre principal point: swapped cx and cy, or width and height in the tile grid or the row stride, move the splat (the frustum guard's use of the size is not engaged, so it is not checked) |
  | `stack` | 50, 33² | five splats on the axis, the front four near-opaque: within about 2 px of the centre the stop at T < 1e-4 leaves the third (centre) or fourth and every later splat unblended |
  | `faint` | 50, 33², 3 cameras (phi 0, 0.4, −1.1) | one anisotropic splat off the axis at opacity 0.332 (below 0.35, so α at 3σ is below 1/255): splat_view's oracle, where the 3σ quad must drop nothing |
  | `faint-overlap` | 50, 48×32, `off-centre`'s camera | three faint splats (opacity 0.27 to 0.33) overlapping, stored middle, front, back: splat_view's back-to-front sort and blend |

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
- **render-gt** gives a scene without images its ground truth, in place: each camera rendered by
  splat_render (`<renderer>` is the binary, or the web build's `.js`, run under `$NODE`) for
  `--frames` (8) frames and its last frame captured, written as the converter writes images
  (`images_<k>.bin`, `images/<name>.png`, the cameras' `imageBatch` and `pixelOffset`, the
  manifest's `files.images` and camera entries) with `source.groundTruth`. `--sh-degree` limits
  the SH bands rendered (default: the scene's), for a target without view-dependent colour. A scene re-fitted to
  these images is fitted to what the renderer draws, so a training test measures the optimizer and
  the backward pass, not a mismatch between the generator's model and the renderer's. It needs a
  GPU, so its tests (`test_render_gt*`) run it with a fake renderer.

## Evaluating a trained scene

`tools/evaluate.py` scores a trained PLY on a scene's held-out views as the reference's
`metrics.py` does: it writes the PLY's splats into a copy of the scene's manifest and cameras,
renders each test camera with splat_render (the binary, or the web build's `.js` under `$NODE`)
and compares each 8-bit capture with the scene's undistorted image by the PSNR and the SSIM
(`tools/splatdata/metrics.py`: the reference's SSIM, as in `make_pair.py`), per view and their
mean, printed and written to `<out dir>/metrics.json`. splat_train's own PSNR scores its float
render, so the two differ a little (within about 0.5 dB, as `check.py` requires).
`--side-by-side <camera> ...` also writes each camera's image beside its render, each scaled to
`--thumbnail` px on its longer side. Its tests
(`tools/tests/test_evaluate.py`) use a fake renderer.

```
tools/evaluate.py build/data/cs5330 <trained.ply> build/splat_render/splat_render <out dir> \
    [--validation] [--side-by-side 0 8] [--thumbnail 256]
```

## GPU runs

Run GPU work (program tests, captures) one at a time on a machine. Locally that's MoltenVK on the
Apple M4; a GPU test is done only when it passes with validation and synchronization validation and
prints no validation message.

A splat_train run that grows the tile entries' capacity isn't bit-deterministic from run to run:
the frame a growth lands on depends on when the asynchronous counts readback arrives. Compare
trained PLYs bit for bit only between runs that don't grow (start them with enough
`--initial-entries`).

## Bumping the wrapper

`external/vulkan_test` is pinned to a commit of github.com/zach-youssef/vulkan_testing. Bump it to
a pushed commit (`git -C external/vulkan_test checkout <sha>`, then commit the submodule), and
read that repository's `docs/consumers.md` and `docs/MIGRATING.md` for what changed.
