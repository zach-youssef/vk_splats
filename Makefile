# Gaussian splatting (Kerbl et al. 2023) in ZGAL. Each program is a ZGAL program,
# programs/<name>/<name>.zgal (importing shared modules from lib/), built with the wrapper's rules
# (zgal.mk) into build/<name>/<name>. Adding a program needs no change here.
#   make           - build every program
#   make test      - the data tools' tests (pytest), the C++ tests (tests/cpp/*.cpp) and every
#                    program's headless test (programs/<name>/test.sh)
#   make web       - build every program for the web into build/web/<name>
#   make test-web  - every program's web test under Node (programs/<name>/test-web.sh)
# docs/building-and-testing.md has the conventions.

# Compiler: clang, the one the wrapper is tested with (Apple's c++ on macOS, clang++ on Linux),
# unless CXX is given in the environment or on the command line (make CXX=g++). make's built-in
# default, or none under make -R, counts as not given.
ifneq ($(filter default undefined,$(origin CXX)),)
CXX = $(if $(filter Darwin,$(ZGAL_OS)),c++,clang++)
endif

# The Vulkan wrapper and ZGAL (git submodule); ZGAL_ROOT=<path> builds with another checkout
ZGAL_ROOT ?= external/vulkan_test

# The platform's include and library paths (the Vulkan SDK's, GLFW's, GLM's) come from zgal.mk's
# ZGAL_CFLAGS and ZGAL_LDFLAGS
CFLAGS = -std=c++20
CXXFLAGS = $(CFLAGS)
LDFLAGS =
LDLIBS = -lvulkan

PYTHON ?= python3

OUT = build

include $(ZGAL_ROOT)/zgal.mk

.DEFAULT_GOAL := all

# ---------------------------------------------------------------------------
# Programs: every programs/<name>/ holding <name>.zgal. They link no libraries beyond Vulkan, so
# headless programs don't need GLFW; a program that needs more (-lglfw for a window) opts in with
# programs/<name>/program.mk setting LIBS_<name>.
# ---------------------------------------------------------------------------
PROGRAMS = $(sort $(foreach dir,$(wildcard programs/*/),$(if $(wildcard $(dir)$(notdir $(patsubst %/,%,$(dir))).zgal),$(notdir $(patsubst %/,%,$(dir))))))

-include $(wildcard programs/*/program.mk)

$(foreach p,$(PROGRAMS),$(eval $(call zgal-program,programs/$(p)/$(p).zgal,$(OUT)/$(p),$(p),$(LIBS_$(p)))))

all: $(foreach p,$(PROGRAMS),$(OUT)/$(p)/$(p))

# For the web (WebGPU): the same programs, compiled to TypeScript and WGSL and built to JavaScript
$(foreach p,$(PROGRAMS),$(eval $(call zgal-web-program,programs/$(p)/$(p).zgal,$(OUT)/web/$(p))))

web: $(foreach p,$(PROGRAMS),$(OUT)/web/$(p))

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# The data tools' tests: test_*.py under tools/ and tests/, in one pytest run, so basenames must be
# unique (there are no __init__.py files)
PY_TESTS = $(shell find tools tests -name 'test_*.py' 2>/dev/null)

test-tools:
ifneq ($(PY_TESTS),)
	$(PYTHON) -m pytest -q $(PY_TESTS)
else
	@echo "test-tools: no test_*.py files"
endif

# C++ tests: each tests/cpp/<name>.cpp is a program (e.g. doctest's, with its own main) built into
# build/tests/<name> and run; doctest and the wrapper's headers are on the include path. Nothing is
# linked, so they test header-only code (the oracles, the wrapper's header-only parts).
CPP_TESTS = $(patsubst tests/cpp/%.cpp,%,$(wildcard tests/cpp/*.cpp))
DOCTEST_CFLAGS = -I$(ZGAL_ROOT)/third_party/doctest

$(OUT)/tests/%: tests/cpp/%.cpp $(wildcard tests/cpp/*.h tests/oracle/*.h)
	@mkdir -p $(dir $@)
	$(CXX) -o $@ $< $(CXXFLAGS) $(ZGAL_CFLAGS) $(DOCTEST_CFLAGS) -Itests

test-cpp: $(foreach t,$(CPP_TESTS),$(OUT)/tests/$(t))
	@set -e; for t in $(CPP_TESTS); do echo "== $$t"; $(OUT)/tests/$$t; done

# Program tests: programs/<name>/test.sh, run from the repository's root as
# test.sh <binary> <output dir>, runs the program headless with --validation
TESTED_PROGRAMS = $(foreach p,$(PROGRAMS),$(if $(wildcard programs/$(p)/test.sh),$(p)))

$(foreach p,$(TESTED_PROGRAMS),$(eval test-$(p): $(OUT)/$(p)/$(p) ; programs/$(p)/test.sh $(OUT)/$(p)/$(p) $(OUT)/test/$(p)))

test-programs: $(foreach p,$(TESTED_PROGRAMS),test-$(p))

test: test-tools test-cpp test-programs

# Web tests: programs/<name>/test-web.sh, run as test-web.sh <main.js> <output dir>, under Node
WEB_TESTED_PROGRAMS = $(foreach p,$(PROGRAMS),$(if $(wildcard programs/$(p)/test-web.sh),$(p)))

$(foreach p,$(WEB_TESTED_PROGRAMS),$(eval test-web-$(p): $(OUT)/web/$(p) ; programs/$(p)/test-web.sh $(OUT)/web/$(p)/$(p).zgal.main.js $(OUT)/test-web/$(p)))

test-web: $(foreach p,$(WEB_TESTED_PROGRAMS),test-web-$(p))

.PHONY: all web test test-tools test-cpp test-programs test-web clean \
	$(foreach p,$(TESTED_PROGRAMS),test-$(p)) $(foreach p,$(WEB_TESTED_PROGRAMS),test-web-$(p))

clean:
	rm -rf $(OUT)
