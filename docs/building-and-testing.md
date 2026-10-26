# Building and testing

## Requirements

The wrapper's: the Vulkan SDK (or the system's Vulkan headers and loader), `glslc`, GLFW, GLM,
Racket 8 and, for the web, Node 24 and npm (`external/vulkan_test/docs/building-and-testing.md`
has the details and the platform paths). The data tools need Python 3 with pytest, numpy and
scipy. Clone with `--recursive`, or run `git submodule update --init --recursive`.

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
working branch, each commit beyond `main` (the newest 20); on `main`, the pushed range.

- `ci.yml`, on every push:
  - `commits` lists the commits (the wrapper's `commits` action, from the pinned submodule);
  - `pending` marks each commit's `test` status pending;
  - `test` runs `test.yml` for them (the newest 50 beyond main);
  - `statuses` posts each commit's `test` status from its job's outcome.
- `test.yml` runs one job per commit (`test <sha>`) on ubuntu-24.04, set up by the wrapper's
  `setup-zgal` action (native toolchain: lavapipe, the Vulkan SDK with its validation layers,
  clang through ccache, Racket and Node), plus pytest, numpy and scipy from Ubuntu's archive (7.4, 1.26 and 1.11 on 24.04). It
  runs `make all web`, then `test-tools`, `test-cpp`, `test-programs` (native, on lavapipe, with
  validation) and `test-web` (Node with Dawn on lavapipe), each running even if an earlier test
  failed, and with `make -k`, so one failing program doesn't stop the others. On failure it uploads `build/test` and `build/test-web`.
- Permissions: the jobs that run code (the repository's, the wrapper's or third-party actions)
  have a read-only `contents: read` token and check out without persisting credentials; only
  `pending` and `statuses`, which check nothing out, can write commit statuses.
- lavapipe's subgroup size is 4, 8 or 16 depending on the runner's CPU, so a kernel that assumes
  one may pass locally and fail here, or the other way round.

## GPU runs

Run GPU work (program tests, captures) one at a time on a machine. Locally that's MoltenVK on the
Apple M4; a GPU test is done only when it passes with validation and synchronization validation and
prints no validation message.

## Bumping the wrapper

`external/vulkan_test` is pinned to a commit of github.com/zach-youssef/vulkan_testing. Bump it to
a pushed commit (`git -C external/vulkan_test checkout <sha>`, then commit the submodule), and
read that repository's `docs/consumers.md` and `docs/MIGRATING.md` for what changed.
