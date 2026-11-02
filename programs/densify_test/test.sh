#!/bin/sh
# densify_test with validation: test.sh <binary> <output dir>. On S3 (make_synthetic.py s3, 2000
# splats) with every 16th splat faded (its opacity logit -7, so the opacity prune has some), from
# camera 0 (scenes.py's arguments: the L2 loss against a smooth target), it checks one
# densification frame:
#   - ample: --max-splats 8192 (every addition fits, none dropped);
#   - tight: --max-splats the splat count (--overflow: some additions must be dropped, among them
#     a split's, which must be kept as it was);
#   - screen: the size prunes (--screen-prune, with the screen-space one at --max-screen-size 5:
#     the 64x64 render's radii stay at or under 8 px, past the reference's 20 never, and it must
#     prune a splat by its radius alone) and the opacity reset (--reset).
# A <binary> ending in .js is the web build (test-web.sh), run under $NODE (node by default).
# $PYTHON (python3 by default) needs numpy, for make_synthetic.py and scenes.py.
set -eu
binary=$1
out=$2
python=${PYTHON:-python3}
here=$(dirname "$0")

rm -rf "$out"
mkdir -p "$out"
"$python" tools/make_synthetic.py s3 "$out/s3" > /dev/null
"$python" -c "
import numpy as np, sys
s = np.fromfile(sys.argv[1], dtype='<f4').reshape(-1, 60)
s[::16, 3] = -7.0
s.tofile(sys.argv[1])" "$out/s3/splats.bin"

program() {  # the native binary, or the web build under Node
    case $binary in
        *.js) "${NODE:-node}" "$binary" "$@" ;;
        *) "$binary" "$@" ;;
    esac
}

status=0
run() {  # run <log name> [extra arguments]
    name=$1
    log="$out/$name.log"
    shift
    # shellcheck disable=SC2046 # the arguments are words without spaces
    if program --validation --frames 200 \
        $("$python" "$here/../grad_test/scenes.py" "$out/s3" 0 "$out/targets.bin") --capacity 1000000 "$@" \
        > "$log" 2>&1; then
        grep '^densify_test:' "$log" >&2
    else
        cat "$log" >&2
        echo "densify_test: $name FAILED" >&2
        status=1
    fi
}

run ample --max-splats 8192
run tight --max-splats 2000 --overflow
run screen --max-splats 8192 --screen-prune --max-screen-size 5 --reset
exit $status
