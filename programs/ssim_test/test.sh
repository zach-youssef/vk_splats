#!/bin/sh
# ssim_test with validation: test.sh <binary> <output dir>. On pairs from make_pair.py (numpy's
# default_rng, so the same on every platform), each against its float64 reference SSIM:
#   - identity (32x32): an image against itself (SSIM 1, every dL/dx 0: --zero-gradients);
#   - constant (32x32): constant images (every interior pixel's S the closed-form luminance term:
#     --interior-ssim), within 2e-4: the variance G*x^2 - mx^2 of a constant x cancels in float to
#     about 1e-7 x^2 (the window's float weights sum to 1 within that), against C2 = 9e-4, so S is
#     off by up to about 1e-4 there (the reference's float32 SSIM is too);
#   - random (40x24, non-square, so a kernel swapping the width and height fails): a correlated
#     random pair, with dL/dx against Richardson-extrapolated central differences on the GPU (32
#     probes, the corners first, h 0.01).
# A <binary> ending in .js is the web build (test-web.sh), run under $NODE (node by default).
# $PYTHON (python3 by default) needs numpy.
set -eu
binary=$1
out=$2
python=${PYTHON:-python3}
here=$(dirname "$0")

rm -rf "$out"
mkdir -p "$out"

program() {  # the native binary, or the web build under Node
    case $binary in
        *.js) "${NODE:-node}" "$binary" "$@" ;;
        *) "$binary" "$@" ;;
    esac
}

status=0
run() {  # run <case> <make_pair.py arguments> -- [ssim_test arguments]
    name=$1
    shift
    pair=
    while [ "$1" != -- ]; do pair="$pair $1"; shift; done
    shift
    log="$out/$name.log"
    # shellcheck disable=SC2086 # the arguments are words without spaces
    args=$("$python" "$here/make_pair.py" "$name" "$out/$name" $pair) || { echo "ssim_test: making $name FAILED" >&2; exit 1; }
    # shellcheck disable=SC2086 # the arguments are words without spaces
    if program --validation --frames 600 $args "$@" > "$log" 2>&1; then
        grep '^ssim_test:' "$log" | grep -v -e ': pixel ' >&2
    else
        cat "$log" >&2
        echo "ssim_test: $name FAILED" >&2
        status=1
    fi
}

run identity -- --zero-gradients
run constant -- --tolerance 2e-4
run random --width 40 --height 24 -- --finite-differences --probes 32 --step 0.01
exit $status
