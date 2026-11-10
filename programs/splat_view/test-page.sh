#!/bin/sh
# splat_view's page in headless Chromium: test-page.sh <web build dir> <output dir> <chromium.ts>.
# It encodes splat_render's committed S3 scene as a stream (the full preset), serves it with S3's
# cameras.bin, and runs the web build's page (splat_view.zgal.html) from camera 0 at a 128x128
# canvas for $frames frames, on SwiftShader, twice:
#   - untouched, whose canvas must match splat_render's golden (golden/s3-camera-0.png) at
#     $minimum dB, as splat_render's page test does;
#   - with W pressed once (chromium.ts --keys w: a keyDown and keyUp at once, as soon as the page
#     reports it's running, polled every 100 ms), which must move the camera (a latched press:
#     docs/architecture.md, "splat_view"), so the canvas must differ from the untouched one by at
#     least $moved_difference/255 at some pixel. The key may land before the app has its camera
#     (the move then lands in the frame it gets it) or after; $frames frames leave room for the
#     poll's delay either way, and the still frames after the move for the projection to show.
# A missing Chromium fails the test rather than skipping it: set CHROMIUM, or run
# npx playwright install chromium-headless-shell. $NODE (node by default) runs the checker,
# $PYTHON (python3 by default) the encoder (with the native library: make native) and the PSNR,
# with numpy and Pillow.
set -eu
web=$1
out=$2
checker=$3
python=${PYTHON:-python3}
here=$(dirname "$0")
golden=$here/../splat_render/golden
frames=48
minimum=45
moved_difference=8  # measured 17/255 (44.4 dB); the untouched canvas is within 1/255 of the golden

rm -rf "$out"
mkdir -p "$out/site/scene"
# The checker serves one directory: the page and the scene, linked or written into it (the page
# fetches the scene's files from the server's root, /scene/...)
ln -s "$(cd "$web" && pwd)" "$out/site/page"
ln -s "$(cd "$golden/s3" && pwd)/cameras.bin" "$out/site/scene/cameras.bin"
"$python" tools/convert.py stream "$golden/s3/splats.bin" "$out/site/scene" \
    --config tools/stream-configs/full.json > "$out/convert.log" ||
    { cat "$out/convert.log"; echo "splat_view: page FAILED, encoding the stream"; exit 1; }
query="arg=/scene/scene.stream&cameras=/scene/cameras.bin&camera=0&frames=$frames&size=128x128"

page() {  # page <name> [checker options]: screenshots $out/<name>.png
    name=$1
    shift
    status=0
    # The explicit timeout leaves a wide margin on CI's SwiftShader
    "${NODE:-node}" "$checker" --serve "$out/site" --timeout 120000 --screenshot "$out/$name.png" "$@" \
        "page/splat_view.zgal.html?$query" > "$out/$name.log" 2>&1 || status=$?
    cat "$out/$name.log"
    if [ $status -eq 3 ]; then
        echo "splat_view: page FAILED, no Chromium (set CHROMIUM, or npx playwright install chromium-headless-shell)"
        exit 1
    elif [ $status -ne 0 ]; then
        echo "splat_view: page $name FAILED"
        exit 1
    fi
}

page still
if ! "$python" "$here/../splat_render/psnr.py" "$out/still.png" "$golden/s3-camera-0.png" $minimum; then
    echo "splat_view: page FAILED, its canvas doesn't match splat_render's golden"
    exit 1
fi
page moved --keys w
# psnr.py fails when a difference exceeds its largest (here one below moved_difference): it must
if "$python" "$here/../splat_render/psnr.py" "$out/moved.png" "$out/still.png" 0 $((moved_difference - 1)); then
    echo "splat_view: page FAILED, pressing W didn't move the camera"
    exit 1
fi
echo "splat_view: page passed, W moved the camera"
