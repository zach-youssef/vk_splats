# Gaussian splatting (Kerbl et al. 2023) in ZGAL. Each program is a ZGAL program,
# programs/<name>/<name>.zgal (importing shared modules from lib/), built with the wrapper's rules
# (zgal.mk) into build/<name>/<name>. Adding a program needs no change here.
#   make           - build every program and the native library (build/native/libsplat_native)
#   make test      - the data tools' tests (pytest), the C++ tests (tests/cpp/*.cpp) and every
#                    program's headless test (programs/<name>/test.sh)
#   make web       - build every program for the web into build/web/<name>
#   make test-web  - every program's web test under Node (programs/<name>/test-web.sh)
#   make test-pages - every program's page test in headless Chromium (programs/<name>/test-page.sh)
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

# zgal.mk's object rules know the generated header but not a program's own headers (e.g.
# render_test's render_check.h, which includes tests/oracle's)
$(foreach p,$(PROGRAMS),$(eval $(patsubst %.cpp,$(OUT)/$(p)/%.o,$(notdir $(wildcard programs/$(p)/*.cpp))): $(wildcard programs/$(p)/*.h tests/oracle/*.h)))

# ---------------------------------------------------------------------------
# The native library: lib/native/c_api.cpp, a C ABI over lib/native/'s headers (on the C++ ZGAL
# generates for lib/native/c_api.zgal), for the data tools' ctypes binding (tools/splatdata/native.py).
# Optimized, since the tools read whole scenes through it. The data tools' tests and the program tests
# (whose checks run the tools) build it first.
# ---------------------------------------------------------------------------
NATIVE_LIB = $(OUT)/native/libsplat_native.$(if $(filter Darwin,$(ZGAL_OS)),dylib,so)
# The binding loads this build of it (another OUT's too), from every recipe's tools
export SPLAT_NATIVE_LIB = $(abspath $(NATIVE_LIB))

$(eval $(call zgal-rule,lib/native/c_api.zgal,$(OUT)/native/gen,$(OUT)/native/spv))

$(NATIVE_LIB): lib/native/c_api.cpp $(wildcard lib/native/*.h) $(OUT)/native/gen/c_api.zgal.stamp
	@mkdir -p $(dir $@)
	$(CXX) -shared -fPIC -fvisibility=hidden -O2 -o $@ $< $(CXXFLAGS) $(ZGAL_CFLAGS) -I$(OUT)/native/gen

native: $(NATIVE_LIB)

all: $(foreach p,$(PROGRAMS),$(OUT)/$(p)/$(p)) $(NATIVE_LIB)

# For the web (WebGPU): the same programs, compiled to TypeScript and WGSL and built to JavaScript
$(foreach p,$(PROGRAMS),$(eval $(call zgal-web-program,programs/$(p)/$(p).zgal,$(OUT)/web/$(p))))

web: $(foreach p,$(PROGRAMS),$(OUT)/web/$(p))

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# The data tools' tests: test_*.py under tools/ and tests/, in one pytest run, so basenames must be
# unique (there are no __init__.py files)
PY_TESTS = $(shell find tools tests -name 'test_*.py' 2>/dev/null)

test-tools: $(NATIVE_LIB)
ifneq ($(PY_TESTS),)
	$(PYTHON) -m pytest -q $(PY_TESTS)
else
	@echo "test-tools: no test_*.py files"
endif

# C++ tests: each tests/cpp/<name>.cpp is a program (e.g. doctest's, with its own main) built into
# build/tests/<name> and run; doctest and the wrapper's headers are on the include path. Nothing is
# linked, so they test header-only code (the oracles, the wrapper's header-only parts, and the C++
# that ZGAL generates for functions). A test with a tests/cpp/<name>.zgal beside it (e.g. importing
# a lib/ module) includes its generated <name>.zgal.h; that file's shaders are compiled too, so
# glslc checks the functions its materials call.
CPP_TESTS = $(patsubst tests/cpp/%.cpp,%,$(wildcard tests/cpp/*.cpp))
CPP_TEST_ZGAL = $(patsubst tests/cpp/%.zgal,%,$(wildcard tests/cpp/*.zgal))
DOCTEST_CFLAGS = -I$(ZGAL_ROOT)/third_party/doctest

$(foreach t,$(CPP_TEST_ZGAL),$(eval $(call zgal-rule,tests/cpp/$(t).zgal,$(OUT)/tests/gen/$(t),$(OUT)/tests/gen/$(t)/spv)))
$(foreach t,$(CPP_TEST_ZGAL),$(eval $(OUT)/tests/$(t): $(OUT)/tests/gen/$(t)/$(t).zgal.stamp))

$(OUT)/tests/%: tests/cpp/%.cpp $(wildcard tests/cpp/*.h tests/oracle/*.h lib/native/*.h)
	@mkdir -p $(dir $@)
	$(CXX) -o $@ $< $(CXXFLAGS) $(ZGAL_CFLAGS) $(DOCTEST_CFLAGS) -Itests \
		$(if $(wildcard tests/cpp/$*.zgal),-I$(OUT)/tests/gen/$*)

test-cpp: $(foreach t,$(CPP_TESTS),$(OUT)/tests/$(t))
	@set -e; for t in $(CPP_TESTS); do echo "== $$t"; $(OUT)/tests/$$t; done

# Program tests: programs/<name>/test.sh, run from the repository's root as
# test.sh <binary> <output dir>, runs the program headless with --validation
TESTED_PROGRAMS = $(foreach p,$(PROGRAMS),$(if $(wildcard programs/$(p)/test.sh),$(p)))

$(foreach p,$(TESTED_PROGRAMS),$(eval test-$(p): $(OUT)/$(p)/$(p) $(NATIVE_LIB) ; programs/$(p)/test.sh $(OUT)/$(p)/$(p) $(OUT)/test/$(p)))

test-programs: $(foreach p,$(TESTED_PROGRAMS),test-$(p))

test: test-tools test-cpp test-programs

# Web-check tools: host programs a web test runs on what a web run wrote, where its checks are C++
# (e.g. render_test's readbacks, checked by the native program's checks). Each
# programs/<name>/web-check/<tool>.cpp (tool names unique across programs) is built like a C++ test
# into build/web-check/<tool>, including its generated <tool>.zgal.h when a <tool>.zgal is beside
# it, and with the program's headers and tests/oracle's as prerequisites.
WEB_CHECK_SOURCES = $(wildcard programs/*/web-check/*.cpp)
WEB_CHECK_TOOL = $(OUT)/web-check/$(basename $(notdir $(1)))
web-check-tools = $(foreach s,$(wildcard programs/$(1)/web-check/*.cpp),$(call WEB_CHECK_TOOL,$(s)))

define web-check-rule
$(if $(wildcard $(1:.cpp=.zgal)),$(call zgal-rule,$(1:.cpp=.zgal),$(OUT)/web-check/gen/$(basename $(notdir $(1))),$(OUT)/web-check/gen/$(basename $(notdir $(1)))/spv))
$(call WEB_CHECK_TOOL,$(1)): $(1) $(wildcard $(dir $(1))*.h $(dir $(1))../*.h tests/oracle/*.h) \
		$(if $(wildcard $(1:.cpp=.zgal)),$(OUT)/web-check/gen/$(basename $(notdir $(1)))/$(basename $(notdir $(1))).zgal.stamp)
	@mkdir -p $$(dir $$@)
	$$(CXX) -o $$@ $$< $$(CXXFLAGS) $$(ZGAL_CFLAGS) -Itests \
		$(if $(wildcard $(1:.cpp=.zgal)),-I$(OUT)/web-check/gen/$(basename $(notdir $(1))))
endef
$(foreach s,$(WEB_CHECK_SOURCES),$(eval $(call web-check-rule,$(s))))

# Web tests: programs/<name>/test-web.sh, run as test-web.sh <main.js> <output dir> <web-check
# dir> under Node, after its program's web build and its web-check tools
WEB_TESTED_PROGRAMS = $(foreach p,$(PROGRAMS),$(if $(wildcard programs/$(p)/test-web.sh),$(p)))

$(foreach p,$(WEB_TESTED_PROGRAMS),$(eval test-web-$(p): $(OUT)/web/$(p) $(call web-check-tools,$(p)) $(NATIVE_LIB) ; programs/$(p)/test-web.sh $(OUT)/web/$(p)/$(p).zgal.main.js $(OUT)/test-web/$(p) $(OUT)/web-check))

test-web: $(foreach p,$(WEB_TESTED_PROGRAMS),test-web-$(p))

# Page tests: programs/<name>/test-page.sh, run as test-page.sh <web build dir> <output dir>
# <chromium.ts> after its program's web build: its page (<name>.zgal.html) in headless Chromium on
# SwiftShader, through the wrapper's checker. A missing Chromium fails them (CHROMIUM names one).
PAGE_TESTED_PROGRAMS = $(foreach p,$(PROGRAMS),$(if $(wildcard programs/$(p)/test-page.sh),$(p)))

$(foreach p,$(PAGE_TESTED_PROGRAMS),$(eval test-pages-$(p): $(OUT)/web/$(p) ; programs/$(p)/test-page.sh $(OUT)/web/$(p) $(OUT)/test-pages/$(p) $(ZGAL_ROOT)/web/tools/chromium.ts))

test-pages: $(foreach p,$(PAGE_TESTED_PROGRAMS),test-pages-$(p))

.PHONY: all native web test test-tools test-cpp test-programs test-web test-pages clean \
	$(foreach p,$(TESTED_PROGRAMS),test-$(p)) $(foreach p,$(WEB_TESTED_PROGRAMS),test-web-$(p)) \
	$(foreach p,$(PAGE_TESTED_PROGRAMS),test-pages-$(p))

clean:
	rm -rf $(OUT)
