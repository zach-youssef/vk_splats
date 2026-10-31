#!/bin/sh
# tile_test on S2, S2 with degenerate splats and S3 (tools/make_synthetic.py), for several cameras
# each, with validation: test.sh <binary> <output dir>. Each run starts with a capacity of 16
# entries, so it also checks an overflowing frame and grows the capacity. Some runs cap the
# dispatches at one or three workgroups, so the kernels' grid-stride loops go round. $PYTHON
# (python3 by default) needs numpy, for make_synthetic.py.
set -eu
binary=$1
out=$2
python=${PYTHON:-python3}
here=$(dirname "$0")

rm -rf "$out"
mkdir -p "$out"
"$python" tools/make_synthetic.py s2 "$out/s2" > /dev/null
"$python" tools/make_synthetic.py s3 "$out/s3" > /dev/null
"$python" "$here/scenes.py" degenerate "$out/s2" "$out/s2-degenerate"

# --timings waits for each frame, so each frame's readbacks arrive before the next frame reuses
# their frame slot. Without it, a frame loop faster than the GPU can drop every copy before it is
# polled, and the read events never come (a vulkan_test gap). Remove it after the vulkan_test
# readback-delivery fix lands (submodule bump).
status=0
run() {  # run <scene> <camera> [<max groups>]
    groups=${3:-65535}
    log="$out/$(basename "$1")-camera-$2-groups-$groups.log"
    # shellcheck disable=SC2046 # the arguments are words without spaces
    if "$binary" --validation --timings --frames 16 --initial-entries 16 --max-groups "$groups" \
        $("$python" "$here/scenes.py" args "$1" "$2") > "$log" 2>&1; then
        grep '^tile_test:' "$log" | tail -n 3
    else
        cat "$log"
        echo "tile_test: $1 camera $2 FAILED"
        status=1
    fi
}

for camera in 0 1; do run "$out/s2" $camera; done
for camera in 0 1; do run "$out/s2-degenerate" $camera; done
for camera in 0 5 13 24; do run "$out/s3" $camera; done
run "$out/s2-degenerate" 1 1
run "$out/s3" 24 1
run "$out/s3" 5 3
exit $status
