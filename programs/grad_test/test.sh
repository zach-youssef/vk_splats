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
#     camera 0 (--samples 24 of each kind: fewer can
#     leave the third band's all negligible);
#   - S2's camera 0 with the L1 loss and the 1D dispatches capped at three workgroups (the
#     grid-stride loops go round), twice, with the hashes compared (deterministic across runs, not
#     only across frames).
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
run "$out/s3" 0 s3-camera-0-parameters 8000 --finite-differences --parameters --samples 24 \
    --capacity 1000000 > /dev/null
first=$(run "$out/s2" 0 s2-l1 16 --l1 --max-groups 3)
second=$(run "$out/s2" 0 s2-l1-again 16 --l1 --max-groups 3)
if [ -z "$first" ] || [ "$first" != "$second" ]; then
    echo "grad_test: S2 camera 0 isn't deterministic: '$first', then '$second'" >&2
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
