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
# Then two runs of the page's opener (fetching the stream: main-browser.ts's streaming asset loader):
#   - partway: $partway_words words a frame (--stream-words) for $partway_frames frames, too few for
#     the stream's 472,432 bytes, so the page must stop still loading, with some base records
#     arrived (decoded and drawn: the Node test, test.sh, checks such frames' images);
#   - refused: a stream whose header has more splats than a stream may hold must fail the page on
#     the opener's refusal (checkStreamHeader, before the graph sizes anything from the header).
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
partway_words=2500  # 10,000 bytes a frame
partway_frames=24   # at most 240,000 bytes

rm -rf "$out"
mkdir -p "$out/site/scene"
# The checker serves one directory: the page and the scene, linked or written into it (the page
# fetches the scene's files from the server's root, /scene/...)
ln -s "$(cd "$web" && pwd)" "$out/site/page"
ln -s "$(cd "$golden/s3" && pwd)/cameras.bin" "$out/site/scene/cameras.bin"
"$python" tools/convert.py stream "$golden/s3/splats.bin" "$out/site/scene" \
    --config tools/stream-configs/full.json > "$out/convert.log" ||
    { cat "$out/convert.log"; echo "splat_view: page FAILED, encoding the stream"; exit 1; }
# The oversized header: StreamHeader.splatCount (at offset 8) streamMaxSplats + 1 (lib/splat_stream.zgal)
"$python" - "$out/site/scene/scene.stream" "$out/site/scene/oversized.stream" <<'EOF'
import struct, sys
with open(sys.argv[1], "rb") as f:
    data = bytearray(f.read())
struct.pack_into("<I", data, 8, 559241)
with open(sys.argv[2], "wb") as f:
    f.write(data)
EOF
query="arg=/scene/scene.stream&cameras=/scene/cameras.bin&camera=0&size=128x128"

run_page() {  # run_page <name> <query> [checker options]: screenshots $out/<name>.png; sets status
    name=$1 page_query=$2
    shift 2
    status=0
    # The explicit timeout leaves a wide margin on CI's SwiftShader
    "${NODE:-node}" "$checker" --serve "$out/site" --timeout 120000 --screenshot "$out/$name.png" "$@" \
        "page/splat_view.zgal.html?$page_query" > "$out/$name.log" 2>&1 || status=$?
    cat "$out/$name.log"
    if [ $status -eq 3 ]; then
        echo "splat_view: page FAILED, no Chromium (set CHROMIUM, or npx playwright install chromium-headless-shell)"
        exit 1
    fi
}

page() {  # page <name> [checker options]: the scene for $frames frames, which must pass
    page_name=$1
    shift
    run_page "$page_name" "$query&frames=$frames" "$@"
    if [ $status -ne 0 ]; then
        echo "splat_view: page $page_name FAILED"
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
echo "splat_view: page: W moved the camera"

run_page partway "$query&frames=$partway_frames&stream-words=$partway_words"
# "streamed A of T bytes, B base and S SH records", the last frame's
base=$(sed -n 's/^.*splat_view: streamed [0-9]* of [0-9]* bytes, \([0-9]*\) base and [0-9]* SH records$/\1/p' \
    "$out/partway.log" | tail -1)
if [ $status -ne 0 ] || ! grep -q 'splat_view: still loading' "$out/partway.log" || [ "${base:-0}" -eq 0 ]; then
    echo "splat_view: page FAILED, not partway through the stream with records arrived ('${base:-}' base)"
    exit 1
fi
echo "splat_view: page: partway through the stream after $partway_frames frames, $base base records arrived"

run_page refused "arg=/scene/oversized.stream&cameras=/scene/cameras.bin&camera=0&size=128x128&frames=$frames"
if [ $status -eq 0 ] || ! grep -q 'checkStreamHeader refuses, code 4 ' "$out/refused.log"; then
    echo "splat_view: page FAILED, the oversized header wasn't refused by the opener"
    exit 1
fi
echo "splat_view: page passed, the oversized header refused by the opener"
