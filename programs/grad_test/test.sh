#!/bin/sh
# grad_test with validation: test.sh <binary> <output dir> [<web-check dir>].
#   - S1's gradient variant against the closed-form oracle, with the L2 loss and the L1;
#   - by finite differences on the GPU (L2 against a smooth pattern; --samples fields of each kind
#     on S3), where the oracle can't reach: S2's cameras (cam_1 80x48, at its targets' offset 4096);
#     S1's one-splat-clamped (alpha's clamp at 0.99) and stack (the stop at T < 1e-4, which
#     --stops requires); S3's cameras 0 and 24 (2000 splats: tiles past one backward batch, which
#     --multi-batch requires);
#   - by finite differences through the splats' parameters (--parameters: the position, the
#     opacity logit, the log-scale, the quaternion and every SH band of degree 3, the DC included),
#     each splat perturbed and projected again: on S2's cameras 0 and 1, and a subset of S3's
#     camera 0 (--samples 24 of each kind: fewer can leave the third band's all negligible);
#   - by finite differences of training's loss, 0.8 L1 + 0.2 D-SSIM (--l1 --dssim 0.2), through
#     the projection's colour: on S2's cameras 0 and 1;
#   - S2's camera 0 with the L1 loss and the 1D dispatches capped at three workgroups (the
#     grid-stride loops go round), twice, with the hashes compared (deterministic across runs, not
#     only across frames);
#   - the backward pass in chunks (--chunk-entries) against one pass: S3's camera 0 (5269 entries,
#     at --capacity 6000) in chunks of 97 entries, 55 of them, so chunk boundaries fall
#     within tiles, and its tiles past one batch (128 entries) span several chunks; and S2's camera 0 (26 entries) as
#     above, in chunks of 5: their hashes must be one pass's, bit for bit;
#   - S2's camera 1 with --target-pixels too small for it, and S3's camera 0 in more than 64
#     chunks (trainChunkPassesMax), which must fail.
# A <binary> ending in .js is the web build (test-web.sh), run under $NODE (node by default): each
# run writes its checked frame's readbacks to <log name>-readbacks ($GRAD_TEST_READBACKS), and
# <web-check dir>/grad_test_check runs the native checks on them; a copy with its gradients
# corrupted must then fail them, so a checker that passes everything can't go unnoticed (the
# finite differences run in the app, the same on both). $PYTHON
# (python3 by default) needs numpy, for make_synthetic.py and scenes.py.
set -eu
binary=$1
out=$2
checks=${3:-}
python=${PYTHON:-python3}
here=$(dirname "$0")

rm -rf "$out"
mkdir -p "$out"
"$python" tools/make_synthetic.py s1 "$out/s1" > /dev/null
"$python" tools/make_synthetic.py s2 "$out/s2" > /dev/null
"$python" tools/make_synthetic.py s3 "$out/s3" > /dev/null

web=
case $binary in *.js)
    web=1
    [ -n "$checks" ] || { echo "grad_test: test.sh needs the web-check directory for a web build" >&2; exit 2; }
    ;;
esac

status=0
program() {  # program <log name> <arguments>: the native binary, or the web build under Node
    readbacks="$out/$1-readbacks"
    shift
    if [ -n "$web" ]; then
        GRAD_TEST_READBACKS=$readbacks "${NODE:-node}" "$binary" "$@" && "$checks/grad_test_check" "$readbacks"
    else
        "$binary" "$@"
    fi
}
run() {  # run <scene> <camera> <log name> <frames> [extra arguments]: the log's hash line on stdout
    scene=$1
    camera=$2
    name=$3
    frames=$4
    log="$out/$name.log"
    shift 4
    # shellcheck disable=SC2046 # the arguments are words without spaces
    if program "$name" --validation --frames "$frames" \
        $("$python" "$here/scenes.py" "$scene" "$camera" "$out/$name-targets.bin") "$@" > "$log" 2>&1; then
        grep '^grad_test:' "$log" | grep -v -e 'hash' -e 'finite difference ' >&2
        grep '^grad_test: hash' "$log" || true
    else
        cat "$log" >&2
        echo "grad_test: $scene camera $camera ($name) FAILED" >&2
        status=1
    fi
}

run "$out/s1/gradient" 0 s1-l2 16 > /dev/null
run "$out/s1/gradient" 0 s1-l1 16 --l1 > /dev/null
for camera in 0 1; do
    run "$out/s2" $camera "s2-camera-$camera-fd" 2000 --finite-differences > /dev/null
done
run "$out/s1/one-splat-clamped" 0 s1-clamped-fd 2000 --finite-differences > /dev/null
run "$out/s1/stack" 0 s1-stack-fd 2000 --finite-differences --stops > /dev/null
for camera in 0 24; do
    run "$out/s3" $camera "s3-camera-$camera-fd" 4000 --finite-differences --samples 24 --multi-batch \
        --capacity 1000000 > /dev/null
done
for camera in 0 1; do
    run "$out/s2" $camera "s2-camera-$camera-parameters" 6000 --finite-differences --parameters > /dev/null
done
for camera in 0 1; do
    run "$out/s2" $camera "s2-camera-$camera-dssim" 2000 --finite-differences --l1 --dssim 0.2 > /dev/null
done
run "$out/s3" 0 s3-camera-0-parameters 8000 --finite-differences --parameters --samples 24 \
    --capacity 1000000 > /dev/null
first=$(run "$out/s2" 0 s2-l1 16 --l1 --max-groups 3)
second=$(run "$out/s2" 0 s2-l1-again 16 --l1 --max-groups 3)
if [ -z "$first" ] || [ "$first" != "$second" ]; then
    echo "grad_test: S2 camera 0 isn't deterministic: '$first', then '$second'" >&2
    status=1
fi
chunked=$(run "$out/s2" 0 s2-l1-chunked 16 --l1 --max-groups 3 --capacity 32 --chunk-entries 5)
if [ -z "$chunked" ] || [ "$first" != "$chunked" ]; then
    echo "grad_test: S2 camera 0 in chunks of 5 entries differs from one pass: '$first', then '$chunked'" >&2
    status=1
fi
whole=$(run "$out/s3" 0 s3-camera-0-whole 16 --capacity 6000)
chunked=$(run "$out/s3" 0 s3-camera-0-chunked 16 --capacity 6000 --chunk-entries 97)
if [ -z "$whole" ] || [ "$whole" != "$chunked" ]; then
    echo "grad_test: S3 camera 0 in chunks of 97 entries differs from one pass: '$whole', then '$chunked'" >&2
    status=1
fi
# A camera whose pixels run past the targets must fail up front (S2's camera 1 starts at pixel
# 4096, which is then the targets' end), not read past them
# shellcheck disable=SC2046 # the arguments are words without spaces
if program short-targets --validation --frames 16 $("$python" "$here/scenes.py" "$out/s2" 1 \
        "$out/short-targets.bin" | sed 's/--target-pixels [0-9]*/--target-pixels 4096/') \
        > "$out/short-targets.log" 2>&1 || ! grep -q "don't fit in --target-pixels" "$out/short-targets.log"; then
    cat "$out/short-targets.log" >&2
    echo "grad_test: FAILED, targets too short for the camera weren't refused" >&2
    status=1
fi
# More backward passes than the graph holds (65 of 100 entries for --capacity 6500, past
# trainChunkPassesMax's 64) must fail up front, not run a frame with entries left out
# shellcheck disable=SC2046 # the arguments are words without spaces
if program many-chunks --validation --frames 16 $("$python" "$here/scenes.py" "$out/s3" 0 \
        "$out/many-chunks.bin") --capacity 6500 --chunk-entries 100 \
        > "$out/many-chunks.log" 2>&1 || ! grep -q "backward passes of --chunk-entries 100, more than 64" "$out/many-chunks.log"; then
    cat "$out/many-chunks.log" >&2
    echo "grad_test: FAILED, more backward passes than the graph holds weren't refused" >&2
    status=1
fi
# Exactly 64 passes (6400 entries of 100) is the most the graph holds, so it must run, which
# pins the check as `>` rather than `>=`
# shellcheck disable=SC2046 # the arguments are words without spaces
if ! program max-chunks --validation --frames 16 $("$python" "$here/scenes.py" "$out/s3" 0 \
        "$out/max-chunks.bin") --capacity 6400 --chunk-entries 100 \
        > "$out/max-chunks.log" 2>&1; then
    cat "$out/max-chunks.log" >&2
    echo "grad_test: FAILED, exactly 64 backward passes weren't accepted" >&2
    status=1
fi
if [ -n "$web" ]; then
    # The checker must fail a frame whose gradients are off (the opacity's raised by 1%)
    corrupt="$out/corrupted-readbacks"
    cp -R "$out/s1-l2-readbacks" "$corrupt"
    "$python" -c "
import numpy as np, sys
g = np.fromfile(sys.argv[1], dtype='<f4').reshape(-1, 12)
g[:, 7] *= 1.01
g.tofile(sys.argv[1])" "$corrupt/gradients.bin"
    if "$checks/grad_test_check" "$corrupt" > "$out/corrupted.log" 2>&1; then
        cat "$out/corrupted.log" >&2
        echo "grad_test: FAILED, grad_test_check passed corrupted gradients" >&2
        status=1
    fi
fi
exit $status
