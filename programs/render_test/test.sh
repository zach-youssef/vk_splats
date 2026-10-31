#!/bin/sh
# render_test with validation: test.sh <binary> <output dir>. Every S1 variant (the closed-form
# oracle, every camera of the rotated ones; off-centre is 48x32 with the principal point off the
# centre), S2's cameras (cam_1 is 80x48: non-square) on a grey background, twice each with the
# hashes compared (deterministic across runs, not only across frames), and S3's cameras 0 and 24
# (160x96), one with the 1D dispatches capped at three workgroups so the tile pipeline's
# grid-stride loops go round under the rasterizer (whose tile-grid dispatch maxGroups doesn't cap).
# $PYTHON (python3 by default) needs numpy, for make_synthetic.py.
set -eu
binary=$1
out=$2
python=${PYTHON:-python3}
here=$(dirname "$0")

rm -rf "$out"
mkdir -p "$out"
"$python" tools/make_synthetic.py s1 "$out/s1" > /dev/null
"$python" tools/make_synthetic.py s2 "$out/s2" > /dev/null
"$python" tools/make_synthetic.py s3 "$out/s3" > /dev/null

status=0
run() {  # run <scene> <camera> <log name> [extra arguments]: the log's hash line on stdout
    scene=$1
    camera=$2
    log="$out/$3.log"
    shift 3
    # shellcheck disable=SC2046 # the arguments are words without spaces
    if "$binary" --validation --frames 16 $("$python" "$here/scenes.py" "$scene" "$camera") "$@" > "$log" 2>&1; then
        grep '^render_test:' "$log" | grep -v 'hash' >&2
        grep '^render_test: hash' "$log"
    else
        cat "$log" >&2
        echo "render_test: $scene camera $camera FAILED" >&2
        status=1
    fi
}

for variant in "$out"/s1/*/; do
    variant=${variant%/}
    cameras=$("$python" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['cameras']))" "$variant/manifest.json")
    camera=0
    while [ "$camera" -lt "$cameras" ]; do
        run "$variant" "$camera" "s1-$(basename "$variant")-camera-$camera" > /dev/null
        camera=$((camera + 1))
    done
done
for camera in 0 1; do
    first=$(run "$out/s2" $camera "s2-camera-$camera" --background-r 0.5 --background-g 0.5 --background-b 0.5)
    second=$(run "$out/s2" $camera "s2-camera-$camera-again" --background-r 0.5 --background-g 0.5 --background-b 0.5)
    if [ -z "$first" ] || [ "$first" != "$second" ]; then
        echo "render_test: S2 camera $camera isn't deterministic: '$first', then '$second'" >&2
        status=1
    fi
done
run "$out/s3" 0 s3-camera-0 > /dev/null
run "$out/s3" 24 s3-camera-24 > /dev/null
run "$out/s3" 24 s3-camera-24-groups-3 --max-groups 3 > /dev/null
exit $status
