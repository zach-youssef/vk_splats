#!/bin/sh
# splat_render's page in headless Chromium: test-page.sh <web build dir> <output dir> <chromium.ts>.
# It runs the web build's own page (splat_render.zgal.html, its present path, which the Node tests
# don't reach) on the committed S3 scene from camera 0 at a 128x128 canvas for 8 frames, on
# SwiftShader, screenshots the canvas and compares it with the native test's golden
# (golden/s3-camera-0.png) at the same PSNR (45 dB). S3 is small (2000 splats), as SwiftShader is
# slow. A missing Chromium fails the test rather than skipping it: set CHROMIUM, or run
# npx playwright install chromium-headless-shell. $NODE (node by default) runs the checker,
# $PYTHON (python3 by default) the PSNR, with numpy and Pillow.
set -eu
web=$1
out=$2
checker=$3
python=${PYTHON:-python3}
here=$(dirname "$0")
minimum=45

rm -rf "$out"
mkdir -p "$out/site"
# The checker serves one directory: the page and the scene, linked into it (the page fetches the
# scene's files from the server's root, /scene/...)
ln -s "$(cd "$web" && pwd)" "$out/site/page"
ln -s "$(cd "$here/golden/s3" && pwd)" "$out/site/scene"

# The page's options from the query string, as test.sh's from scene_args.py (paths made /scene/...)
query=$("$python" tools/scene_args.py --query /scene "$here/golden/s3" 0)

log="$out/s3-camera-0.log"
status=0
# The page takes a few seconds on CI's SwiftShader (about 10 s locally); the explicit timeout leaves
# a wide margin without relying on the checker's default
"${NODE:-node}" "$checker" --serve "$out/site" --timeout 120000 --screenshot "$out/s3-camera-0.png" \
    "page/splat_render.zgal.html?$query&frames=8&size=128x128" > "$log" 2>&1 || status=$?
cat "$log"
if [ $status -eq 3 ]; then
    echo "splat_render: page FAILED, no Chromium (set CHROMIUM, or npx playwright install chromium-headless-shell)"
    exit 1
elif [ $status -ne 0 ]; then
    echo "splat_render: page FAILED"
    exit 1
fi
if ! "$python" "$here/psnr.py" "$out/s3-camera-0.png" "$here/golden/s3-camera-0.png" $minimum; then
    echo "splat_render: page FAILED, its canvas doesn't match the golden"
    exit 1
fi
