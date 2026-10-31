#!/bin/sh
# splat_render's regression test, headless with validation: test.sh <binary> <output dir>. It
# renders the committed S3 scene (golden/s3, tools/make_synthetic.py s3 with seed 0, committed so
# a change in numpy's random streams can't move it) from camera 0 (128x128) and camera 24
# (160x96), captures the last frame, and compares it with the reviewed golden capture
# (golden/s3-camera-<k>.png) at a PSNR of at least 45 dB. Camera 24 starts with a capacity of 16
# entries, so the program must grow it before the frame it captures. $PYTHON (python3 by default)
# needs numpy and Pillow.
set -eu
binary=$1
out=$2
python=${PYTHON:-python3}
here=$(dirname "$0")
minimum=45

rm -rf "$out"
mkdir -p "$out"

status=0
render() {  # render <camera> [extra arguments]
    camera=$1
    shift
    log="$out/s3-camera-$camera.log"
    # shellcheck disable=SC2046 # the arguments are words without spaces
    if "$binary" --validation --frames 8 --capture "$out/s3-camera-$camera.png" \
        $("$python" tools/scene_args.py "$here/golden/s3" "$camera") "$@" > "$log" 2>&1 &&
        "$python" "$here/psnr.py" "$out/s3-camera-$camera.png" "$here/golden/s3-camera-$camera.png" $minimum; then
        :
    else
        cat "$log"
        echo "splat_render: S3 camera $camera FAILED"
        status=1
    fi
}

render 0
render 24 --initial-entries 16
if ! grep -q 'growing it' "$out/s3-camera-24.log"; then
    cat "$out/s3-camera-24.log"
    echo "splat_render: FAILED, camera 24 didn't grow its capacity from 16 entries"
    status=1
fi
exit $status
