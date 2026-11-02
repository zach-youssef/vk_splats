#!/bin/sh
# splat_train's regression test, headless with validation: test.sh <binary> <output dir>. It gives
# the committed S3 scene (programs/splat_render/golden/s3) its ground truth with splat_render
# (make_synthetic.py render-gt, the same build as <binary>: native, or the web build for a .js),
# with SH degree 0: 22 training views can't pin down the random degree-3 colours well enough for
# the held-out views in a short run. It perturbs the splats (make_synthetic.py perturb, seed 0)
# and fits them back with the reference's learning rates for 2000 iterations at 128x128 (and the
# 160x96 view), the SH degree rising every 500. It passes when the held-out PSNR is at least
# 30 dB (32.0 on MoltenVK; the start is 20), the training loss (L1) falls at least 5 times from the
# first epoch to the last (7.7), nothing is NaN, and the exported PLY, rendered again by
# splat_render from each held-out camera, scores what splat_train reported (check.py). Training
# converges slowly from perturb's start: 300 iterations reach about 26 dB. The tile entries'
# capacity starts at 256, so the first iterations overflow and grow it: check.py requires skipped
# iterations (2 or 3), which covers Adam's skip, Record's -1 and their report. $PYTHON (python3 by default)
# needs numpy and Pillow.
set -eu
binary=$1
out=$2
python=${PYTHON:-python3}
here=$(dirname "$0")
iterations=2000
# splat_render's build beside this one: build/splat_train/splat_train or
# build/web/splat_train/splat_train.zgal.main.js
renderer=$(printf '%s\n' "$binary" | sed 's/splat_train/splat_render/g')

rm -rf "$out"
mkdir -p "$out"

program() {  # the native binary, or the web build under Node
    case $binary in
        *.js) "${NODE:-node}" "$binary" "$@" ;;
        *) "$binary" "$@" ;;
    esac
}

fail() {
    [ -f "$out/train.log" ] && cat "$out/train.log"
    echo "splat_train: FAILED, $1"
    exit 1
}

cp -R programs/splat_render/golden/s3 "$out/gt"
"$python" tools/make_synthetic.py render-gt "$out/gt" "$renderer" --validation --sh-degree 0 ||
    fail "rendering the ground truth"
"$python" tools/make_synthetic.py perturb "$out/gt" "$out/start" --seed 0 > /dev/null ||
    fail "perturbing the scene"
# The loading frames, the iterations, the held-out views and the readback's latency
# shellcheck disable=SC2046 # the arguments are words without spaces
SPLAT_TRAIN_PLY="$out/trained.ply" program --validation --frames $((iterations + 64)) \
    $("$python" tools/scene_args.py --train "$out/start") \
    --iterations $iterations --sh-interval 500 --initial-entries 256 --max-splats 4096 > "$out/train.log" 2>&1 ||
    fail "the run exited with $?"
"$python" "$here/check.py" "$out" "$renderer" --min-psnr 30 --min-ratio 5 --min-skipped 1 || fail "the check"
