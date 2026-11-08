#!/bin/sh
# stream_test with validation: test.sh <binary> <output dir>. Each case writes the fixture
# (tools/tests/stream_fixture.py, at SH degree 3, with the encodings' edge values: 300 splats but
# where said), encodes
# it with `convert.py stream` and a config, and runs make_case.py for the reference and stream_test's
# arguments; the stream then arrives --chunk bytes a frame:
#   - the presets (full, f16, web) and configs/ (u16 at degree 1, u8 with explicit ranges at degree 2,
#     mixed at degree 0), 1000 bytes a frame (most frames end inside a record);
#   - full in one frame (all 300 records in one dispatch: several workgroups) and web 4 bytes a frame
#     (20 splats);
#   - refused headers, patched (by make_case.py --set) into the f16 stream, each with its code: a bad
#     magic, version, SH degree, stride, offset, encoding, component count, shared range and field
#     layout (checkStreamHeader's codes 1 to 10), the cap (splatCount 559,241), and a
#     header at the cap (559,240) that passes the check but not the 300-splat working buffer
#     (streamOverCapacity);
#   - and two that must fail on a MISMATCH: the f16 case expecting a refusal, and with a reference
#     bit flipped.
# A <binary> ending in .js is the web build (test-web.sh), run under $NODE (node by default).
# $PYTHON (python3 by default) needs numpy, and the native library (make native) for convert.py.
set -eu
binary=$1
out=$2
python=${PYTHON:-python3}
here=$(dirname "$0")
tools=$here/../../tools

rm -rf "$out"
mkdir -p "$out"

program() {  # the native binary, or the web build under Node
    case $binary in
        *.js) "${NODE:-node}" "$binary" "$@" ;;
        *) "$binary" "$@" ;;
    esac
}

status=0
run() {  # run <case> <splats> <config> <chunk> <expected code> [make_case.py arguments]
    name=$1 count=$2 config=$3 chunk=$4 code=$5
    shift 5
    dir="$out/$name"
    log="$out/$name.log"
    "$python" "$tools/tests/stream_fixture.py" "$dir" --count "$count" --seed 7 &&
        "$python" "$tools/convert.py" stream "$dir/splats.bin" "$dir" --config "$config" > "$dir/convert.log" &&
        args=$("$python" "$here/make_case.py" "$dir" "$@") || { echo "stream_test: making $name FAILED" >&2; exit 1; }
    # shellcheck disable=SC2086 # the arguments are words without spaces
    if program --validation --frames 2400 $args --chunk "$chunk" --expect-code "$code" > "$log" 2>&1; then
        grep '^stream_test:' "$log" | sed "s/^stream_test:/stream_test: $name:/" >&2
    else
        cat "$log" >&2
        echo "stream_test: $name FAILED" >&2
        status=1
    fi
}

presets=$tools/stream-configs
configs=$here/configs
run full 300 "$presets/full.json" 1000 0
run f16 300 "$presets/f16.json" 1000 0
run web 300 "$presets/web.json" 1000 0
run u16-degree1 300 "$configs/u16-degree1.json" 1000 0
run u8-degree2 300 "$configs/u8-degree2.json" 1000 0
run f16-degree0 300 "$configs/f16-degree0.json" 1000 0
run full-at-once 300 "$presets/full.json" 1000000 0
run web-by-word 20 "$presets/web.json" 4 0
# Refused headers: checkStreamHeader's codes (lib/splat_stream.zgal), then streamOverCapacity (11)
f16=$presets/f16.json
run bad-magic 300 "$f16" 1000 1 --refused --set magic=1397510228
run bad-version 300 "$f16" 1000 2 --refused --set version=2
run bad-degree 300 "$f16" 1000 3 --refused --set shDegree=4
run over-cap 300 "$f16" 1000 4 --refused --set splatCount=559241
run bad-stride 300 "$f16" 1000 5 --refused --set baseStride=38
run bad-offset 300 "$f16" 1000 6 --refused --set baseOffset=436
run bad-encoding 300 "$f16" 1000 7 --refused --set fields.2.encoding=4
run bad-components 300 "$f16" 1000 8 --refused --set fields.0.components=2
run bad-shared-range 300 "$f16" 1000 9 --refused --set fields.5.sharedRange=2
run bad-field-layout 300 "$f16" 1000 10 --refused --set fields.1.offset=0
run at-cap 300 "$f16" 1000 11 --refused --set splatCount=559240 --set shOffset=20133072
# The checks must catch a wrong decode: the f16 case again, expecting a refusal, and with one
# reference value's lowest bit flipped (splat 150's opacity, an f16 field: bit for bit).
fails() {  # fails <case> <directory of a passed case> <expected code> <flip a bit: 0 or 1>
    name=$1 dir=$2 code=$3 flip=$4
    args=$("$python" "$here/make_case.py" "$dir") &&
        if [ "$flip" = 1 ]; then
            "$python" -c 'import sys, numpy as np
v = np.fromfile(sys.argv[1], dtype=np.uint32); v[150 * 60 + 3] ^= 1; v.tofile(sys.argv[1])' "$dir/reference.bin"
        fi || { echo "stream_test: making $name FAILED" >&2; exit 1; }
    # shellcheck disable=SC2086 # the arguments are words without spaces
    # Failing isn't enough: it must be a check's MISMATCH, not a crash, a validation message or the
    # frames running out
    if program --validation --frames 2400 $args --chunk 1000 --expect-code "$code" > "$out/$name.log" 2>&1; then
        cat "$out/$name.log" >&2
        echo "stream_test: $name passed, but must fail: FAILED" >&2
        status=1
    elif grep -q MISMATCH "$out/$name.log"; then
        echo "stream_test: $name: failed on a MISMATCH, as it must" >&2
    else
        cat "$out/$name.log" >&2
        echo "stream_test: $name failed, but not on a MISMATCH: FAILED" >&2
        status=1
    fi
}
fails wrong-code "$out/f16" 1 0
fails flipped-bit "$out/f16" 0 1
exit $status
