#!/bin/sh
# splat_view's regression test, headless with validation: test.sh <binary> <output dir> <web-check
# dir> (view_oracle_check, built by make test-splat_view or test-web-splat_view). It encodes
# splat_render's S3 scene (../splat_render/golden/s3) as streams with the full and web presets
# (tools/convert.py stream), and for each:
#   - renders cameras 0 (128x128) and 24 (160x96) from --cameras at their own --size, with
#     --inspect (the image must be Encode's of accum, and every splat decoded, sorted and drawn), and
#     compares each capture with splat_render's golden (the tile renderer's) at a PSNR of at least
#     $golden_minimum dB and no difference above $largest_difference/255: the viewer blends back to
#     front in rgba16f with no early stop, so it isn't bit for bit (docs/architecture.md);
#   - and the web preset's captures with the full preset's at least $preset_minimum dB, held to the
#     same largest difference.
# The PSNR alone can't catch a lost splat (one costs 51-75 dB); the largest difference does (one
# costs 15/255 or more on these cameras: docs/architecture.md, "splat_view"). Each comparison
# prints both, so a log shows the margin.
# Then the start camera without --cameras must be the header's: the stream's header camera (bytes
# 32 to 144) written as a one-camera cameras.bin, rendered with and without --cameras at its size,
# gives the same capture, with splats in it (--inspect: some pixel covered). And a --splat-count
# that isn't the header's fails.
# Last, the closed-form oracle (below, and docs/architecture.md, "splat_view").
# A <binary> ending in .js is the web build (test-web.sh), run under $NODE (node by default), held to
# the same bars. $PYTHON (python3 by default) needs numpy and Pillow, and the native library (make
# native) for convert.py.
set -eu
binary=$1
out=$2
checks=$3
python=${PYTHON:-python3}
here=$(dirname "$0")
tools=$here/../../tools
golden=$here/../splat_render/golden
golden_minimum=50  # measured 59.0-59.9 dB (Apple M-series, MoltenVK), largest difference 2/255
preset_minimum=50  # measured 61.9 dB, largest difference 1/255
largest_difference=8  # in 1/255 units, for both

rm -rf "$out"
mkdir -p "$out"

program() {  # the native binary, or the web build under Node
    case $binary in
        *.js) "${NODE:-node}" "$binary" "$@" ;;
        *) "$binary" "$@" ;;
    esac
}

status=0
fail() {  # fail <log> <message>
    cat "$1"
    echo "splat_view: $2 FAILED"
    status=1
}

view() {  # view <name> <stream> [arguments]: writes $out/<name>.png
    name=$1 stream=$2
    shift 2
    # shellcheck disable=SC2046 # the arguments are words without spaces
    program --validation --frames 16 --capture "$out/$name.png" \
        $("$python" "$tools/scene_args.py" --view "$stream") "$@" > "$out/$name.log" 2>&1
}

psnr() {  # psnr <capture> <reference> <minimum> <log>: prints the PSNR and largest difference
    "$python" "$here/../splat_render/psnr.py" "$1" "$2" "$3" $largest_difference >> "$4" 2>&1 &&
        tail -1 "$4"
}

for preset in full web; do
    "$python" "$tools/convert.py" stream "$golden/s3/splats.bin" "$out/$preset" \
        --config "$tools/stream-configs/$preset.json" > "$out/$preset-convert.log" ||
        { cat "$out/$preset-convert.log"; echo "splat_view: encoding $preset FAILED"; exit 1; }
    for camera in "0 128x128" "24 160x96"; do
        # shellcheck disable=SC2086 # the camera index and its size
        set -- $camera
        name=$preset-camera-$1
        if view "$name" "$out/$preset/scene.stream" --cameras "$golden/s3/cameras.bin" --camera "$1" --size "$2" --inspect &&
            grep -q '^splat_view: passed' "$out/$name.log" &&
            psnr "$out/$name.png" "$golden/s3-camera-$1.png" $golden_minimum "$out/$name.log"; then
            grep '^splat_view: inspect' "$out/$name.log"
        else
            fail "$out/$name.log" "$preset camera $1 against splat_render's golden"
        fi
    done
done
for camera in 0 24; do
    log=$out/presets-camera-$camera.log
    : > "$log"
    psnr "$out/web-camera-$camera.png" "$out/full-camera-$camera.png" $preset_minimum "$log" ||
        fail "$log" "web against full preset, camera $camera"
done

# The header's camera: without --cameras, and from a cameras.bin holding just it, at its size
size=$("$python" - "$out/full/scene.stream" "$out/header-camera.bin" <<'EOF'
import struct, sys
with open(sys.argv[1], "rb") as f:
    header = f.read(144)
with open(sys.argv[2], "wb") as f:
    f.write(header[32:144])  # StreamHeader.camera: a Camera (112 bytes) at offset 32
print("%dx%d" % struct.unpack_from("<2I", header, 32 + 96))  # Camera.size, at its offset 96
EOF
)
if view header "$out/full/scene.stream" --size "$size" --inspect &&
    grep -q '^splat_view: passed' "$out/header.log" &&
    view header-file "$out/full/scene.stream" --cameras "$out/header-camera.bin" --size "$size" &&
    cmp -s "$out/header.png" "$out/header-file.png"; then
    echo "splat_view: the start camera is the header's ($size)"
else
    cat "$out/header.log" "$out/header-file.log"
    echo "splat_view: the start camera without --cameras FAILED: not the header's"
    status=1
fi

# A --splat-count that isn't the header's must fail (as must the header checks behind it)
log=$out/wrong-count.log
if program --validation --frames 16 "$out/full/scene.stream" --splat-count 1999 \
    --stream-bytes "$(wc -c < "$out/full/scene.stream" | tr -d " ")" > "$log" 2>&1; then
    fail "$log" "a wrong --splat-count passing"
elif grep -q 'splats, not --splat-count' "$log"; then
    echo "splat_view: a wrong --splat-count refused"
else
    fail "$log" "a wrong --splat-count (not refused for it)"
fi
# The closed-form oracle: S1's faint variants (tools/make_synthetic.py: opacity below 0.35 and at
# most 3 splats, so the 3-sigma quad drops nothing the oracle blends) as full-preset (f32) streams,
# every camera at its size with --inspect, accum written to a file ($SPLAT_VIEW_ACCUM) and checked
# by <web-check dir>/view_oracle_check against the oracle within 2e-3. Then a copy of one with a
# pixel's red moved by 4e-3 must fail the check, so it isn't vacuous.
for variant in faint faint-overlap; do
    scene=$out/s1/$variant
    "$python" "$tools/make_synthetic.py" s1 "$out/s1" --variant $variant > /dev/null &&
        "$python" "$tools/convert.py" stream "$scene/splats.bin" "$scene/stream" \
            --config "$tools/stream-configs/full.json" --cameras "$scene/cameras.bin" > "$out/$variant-convert.log" ||
        { cat "$out/$variant-convert.log"; echo "splat_view: encoding S1 $variant FAILED"; exit 1; }
    # Each camera's index and size (Camera.size at its offset 96 of 112 bytes)
    "$python" - "$scene/cameras.bin" > "$out/$variant-sizes.txt" <<'EOF'
import struct, sys
with open(sys.argv[1], "rb") as f:
    data = f.read()
for k in range(len(data) // 112):
    print("%d %dx%d" % ((k,) + struct.unpack_from("<2I", data, 112 * k + 96)))
EOF
    while read -r camera size; do
        name=oracle-$variant-camera-$camera
        # (a subshell: an assignment before a function call may outlive it)
        if (SPLAT_VIEW_ACCUM=$out/$name.accum && export SPLAT_VIEW_ACCUM &&
            view "$name" "$scene/stream/scene.stream" --cameras "$scene/cameras.bin" --camera "$camera" \
                --size "$size" --inspect) &&
            grep -q '^splat_view: passed' "$out/$name.log" &&
            "$checks/view_oracle_check" "$scene/splats.bin" "$scene/cameras.bin" "$camera" "$out/$name.accum" \
                >> "$out/$name.log" 2>&1; then
            grep '^view_oracle_check: camera' "$out/$name.log"
        else
            fail "$out/$name.log" "S1 $variant camera $camera against the closed-form oracle"
        fi
    done < "$out/$variant-sizes.txt"
done
log=$out/oracle-corrupted.log
"$python" - "$out/oracle-faint-overlap-camera-0.accum" "$out/oracle-corrupted.accum" <<'EOF'
import sys
import numpy as np
accum = np.fromfile(sys.argv[1], dtype=np.float16).reshape(-1, 4).copy()
pixel = int(np.argmax(accum[:, 3]))  # the most covered: far from any threshold
accum[pixel, 0] = np.float16(float(accum[pixel, 0]) + 4e-3)
accum.tofile(sys.argv[2])
EOF
if "$checks/view_oracle_check" "$out/s1/faint-overlap/splats.bin" "$out/s1/faint-overlap/cameras.bin" 0 \
    "$out/oracle-corrupted.accum" > "$log" 2>&1; then
    fail "$log" "a corrupted accum passing the oracle"
elif grep -q 'MISMATCH' "$log"; then
    echo "splat_view: a corrupted accum fails the oracle"
else
    fail "$log" "a corrupted accum (not failed for a mismatch)"
fi
# And the bounds the oracle relies on are enforced, not assumed: a fixture whose first splat has
# opacity 0.5 (its logit set to 0), and one with every splat twice (6 blends where the 3 overlap)
# must each be refused (given camera 0's accum, of the right size; it's never compared)
"$python" - "$out/s1/faint-overlap/splats.bin" "$out" <<'EOF'
import struct, sys
with open(sys.argv[1], "rb") as f:
    data = f.read()
# A std430 Splat is 240 bytes, its opacity logit a float at offset 12 (tools/splatdata/splats.py)
with open(sys.argv[2] + "/opaque-splats.bin", "wb") as f:
    f.write(data[:12] + struct.pack("<f", 0.0) + data[16:])
with open(sys.argv[2] + "/doubled-splats.bin", "wb") as f:
    f.write(data + data)
EOF
for refused in "opaque:isn't below 0.35" "doubled:blends [0-9]* splats"; do
    fixture=${refused%%:*}
    log=$out/oracle-$fixture.log
    if "$checks/view_oracle_check" "$out/$fixture-splats.bin" "$out/s1/faint-overlap/cameras.bin" 0 \
        "$out/oracle-faint-overlap-camera-0.accum" > "$log" 2>&1; then
        fail "$log" "the $fixture fixture passing the oracle"
    elif grep -q "${refused#*:}" "$log"; then
        echo "splat_view: the $fixture fixture is refused by the oracle"
    else
        fail "$log" "the $fixture fixture (not refused for its bound)"
    fi
done
# The base speed (docs/architecture.md, "Controls") comes from the cameras' spread, or from the
# start camera's distance when they don't spread. S3's 25 cameras spread ~4 (speed ~1: a quarter),
# and the moves runs below log it. A header-only run takes the header camera's distance, as
# does a cameras file of just it; a file of one camera, and of coincident centres (camera 0 seven
# times, whose float32 spread is a rounding residue, not 0), must take the same speed from the
# fallback (about 1 here), not ~1e-8.
speed_of() {  # speed_of <log>: the base speed it printed
    sed -n 's/^splat_view: base speed \([^ ]*\) .*/\1/p' "$1"
}
head -c 112 "$golden/s3/cameras.bin" > "$out/one-camera.bin"
: > "$out/coincident.bin"
for _ in 1 2 3 4 5 6 7; do cat "$out/one-camera.bin" >> "$out/coincident.bin"; done
view coincident "$out/full/scene.stream" --cameras "$out/coincident.bin" --size 128x128 ||
    fail "$out/coincident.log" "the coincident cameras run"
view one-camera "$out/full/scene.stream" --cameras "$out/one-camera.bin" --size 128x128 ||
    fail "$out/one-camera.log" "the one camera run"
header_speed=$(speed_of "$out/header.log")
if [ -n "$header_speed" ] && grep -q "^splat_view: base speed .*(from the header's camera)" "$out/header.log" &&
    [ "$(speed_of "$out/header-file.log")" = "$header_speed" ] &&
    [ "$(speed_of "$out/coincident.log")" = "$(speed_of "$out/one-camera.log")" ] &&
    awk -v s="$(speed_of "$out/coincident.log")" 'BEGIN { exit !(s > 0.1) }' &&
    grep -q "^splat_view: base speed .*(from the start camera's distance: the 7 cameras" "$out/coincident.log" &&
    grep -q "^splat_view: base speed 0\.9[0-9]* (the 25 cameras spread 3\.9" "$out/full-camera-0.log"; then
    echo "splat_view: base speed: the fallback for no, one and coincident cameras, the spread for S3's"
else
    cat "$out/header.log" "$out/header-file.log" "$out/one-camera.log" "$out/coincident.log"
    echo "splat_view: the base speed FAILED: its source or value"
    status=1
fi
# The controls (docs/architecture.md, "splat_view"): Project and the sort run only when the camera
# changes. The still runs above projected once (--inspect counts the GPU's runs and checks them
# against the app's). --scripted-moves 6 shrinks the surface (resize!) the frame after the app is
# ready, then moves (W and a drag, through the controls' code) for 6 frames, then is still: Project
# must run in 2 + 6 frames (ready, the shrink and the moves), not after, and at 2 and 3 frames in
# flight the last, still frame (after a move) must be the same image, byte for byte, as a run
# projecting in every frame (a fresh render of the same pose and size).
for name in full-camera-0 full-camera-24 web-camera-0 web-camera-24 header; do
    grep -q '^splat_view: inspect: Project ran in 1 frames' "$out/$name.log" ||
        fail "$out/$name.log" "$name: Project running on a still camera"
done
moves=6
script() {  # script <log> [arguments]: the moves on the full preset's camera 0, 40 frames
    log=$1
    shift
    # shellcheck disable=SC2046 # the arguments are words without spaces
    program --validation --frames 40 $("$python" "$tools/scene_args.py" --view "$out/full/scene.stream") \
        --cameras "$golden/s3/cameras.bin" --camera 0 --size 128x128 --scripted-moves $moves "$@" \
        > "$log" 2>&1 || fail "$log" "$log's run"
}
for frames_in_flight in 2 3; do
    name=moves-$frames_in_flight
    script "$out/$name-moves.log" --frames-in-flight $frames_in_flight --capture "$out/$name-moves.png" --inspect
    script "$out/$name-fresh.log" --frames-in-flight $frames_in_flight --capture "$out/$name-fresh.png" \
        --project-every-frame
    if ! grep -q "^splat_view: inspect: Project ran in $((moves + 2)) frames" "$out/$name-moves.log"; then
        fail "$out/$name-moves.log" "$name: Project not running in exactly $((moves + 2)) frames"
    elif ! cmp -s "$out/$name-moves.png" "$out/$name-fresh.png"; then
        fail "$out/$name-moves.log" "$name: the still frame after moving against a fresh render"
    else
        grep '^splat_view: inspect' "$out/$name-moves.log"
        echo "splat_view: $name: still after moving, the image is a fresh render's"
    fi
done
# The sort must run in the frames Project does (the same #:enabled flag), not every frame: the
# native --timings table counts the frames each node ran in (the web has no --timings). Sorting
# sorted keys changes nothing, so no image or count above would catch an always-on sort.
case $binary in
    *.js) ;;
    *)
        log=$out/sort-frames.log
        script "$log" --timings
        project=$(awk '$1 == "project" { print $NF }' "$log")
        sort=$(awk '$1 == "sortSplats" { print $NF }' "$log")
        if [ "$project" = $((moves + 2)) ] && [ "$sort" = $((moves + 2)) ]; then
            echo "splat_view: Project and the sort ran in $((moves + 2)) of 40 frames"
        else
            fail "$log" "Project in '$project' frames and the sort in '$sort', not $((moves + 2)):"
        fi
        ;;
esac
exit $status
