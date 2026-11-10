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
# gives the same capture, with splats in it (--inspect: some pixel covered). Then the opener (native
# and web) must refuse a header checkStreamHeader refuses (too many splats; a splat count whose
# stream size wraps a uint) and a stream ending within its header, failing before the first frame,
# before the graph sizes anything from the header. And natively (the web opener reads streams only)
# S3 as an INRIA PLY (which the native opener encodes in full, f32) renders bit for bit as its
# full-preset stream, from --cameras and from the header's camera.
# Then streaming (below): a frame partway through the stream draws the scene its arrived bytes
# hold, with the decode checked in every frame (--check-decode).
# Then the closed-form oracle (below, and docs/architecture.md, "splat_view").
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

view() {  # view <name> <scene> [arguments]: writes $out/<name>.png
    name=$1 scene=$2
    shift 2
    program --validation --frames 16 --capture "$out/$name.png" "$scene" "$@" > "$out/$name.log" 2>&1
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

# Headers the opener must refuse, so the graph never sizes its buffers from them: one with more splats
# than a stream may hold (streamMaxSplats + 1, lib/splat_stream.zgal), one with 2^30 splats and its
# shOffset where it must be at that count in a uint (432: the strides' products are multiples of
# 2^32), whose stream-buffer would size to 108 words, and a stream ending within its 432-byte header.
# Each must fail before the first frame on the opener's refusal (the app's later check, on the header
# PlanDecode copies out, prints another message: a run passing the header on, as the web's plain
# loadAssetStream did, fails this).
"$python" - "$out/full/scene.stream" "$out" <<'EOF'
import struct, sys
with open(sys.argv[1], "rb") as f:
    data = f.read()
oversized = bytearray(data)
struct.pack_into("<I", oversized, 8, 559241)  # StreamHeader.splatCount, at offset 8: streamMaxSplats + 1
wrapping = bytearray(data)
struct.pack_into("<I", wrapping, 8, 1 << 30)
struct.pack_into("<I", wrapping, 28, (432 + (1 << 30) * 56) % (1 << 32))  # shOffset, at offset 28: 432
for name, contents in (("oversized", oversized), ("wrapping", wrapping), ("short", data[:400])):
    with open("%s/%s.stream" % (sys.argv[2], name), "wb") as f:
        f.write(contents)
EOF
for refused in "oversized:checkStreamHeader refuses, code 4 " "wrapping:checkStreamHeader refuses, code 4 " \
    "short:ends within its 432-byte stream header (400 bytes)"; do
    case_name=${refused%%:*}
    log=$out/$case_name.log
    if program --validation --frames 16 "$out/$case_name.stream" > "$log" 2>&1; then
        fail "$log" "the $case_name header passing"
    elif grep -qF "${refused#*:}" "$log"; then
        echo "splat_view: the $case_name header refused by the opener"
    else
        fail "$log" "the $case_name header (not refused by the opener)"
    fi
done

case $binary in
    *.js) ;;
    *)
        # A PLY, encoded by the opener, against its full-preset stream: bit for bit, from --cameras
        # and from the header's camera (the encoder's default camera, the same for both)
        "$python" - "$tools" "$golden/s3/splats.bin" "$out/s3.ply" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from splatdata.splats import from_std430, write_ply
with open(sys.argv[2], "rb") as f:
    write_ply(sys.argv[3], from_std430(f.read()))
EOF
        for args in "--cameras $golden/s3/cameras.bin --camera 0 --size 128x128" "--size $size"; do
            # shellcheck disable=SC2086 # the arguments are words without spaces
            if view ply "$out/s3.ply" $args && view ply-stream "$out/full/scene.stream" $args &&
                cmp -s "$out/ply.png" "$out/ply-stream.png"; then
                echo "splat_view: the PLY renders as its stream ($args)"
            else
                cat "$out/ply-stream.log"
                fail "$out/ply.log" "the PLY against its stream ($args)"
            fi
        done
        ;;
esac

# Streaming: --stream-words small enough that a frame's stream-buffer holds part of the stream. At
# 2 and 3 frames in flight the last frame of each run, partway through, must be the image, byte for
# byte, of the scene its arrived bytes hold (the first B splats, with the SH above degree 0 of only
# the first S: the log's "B base and S SH records") encoded whole, and when that is the whole base
# without SH (B 2000, S 0), S3 encoded at SH degree 0. Natively the stream-buffer gives the header
# alone in frame 0 (zgal::openStream reads it first), then that many words a frame, so frame f holds
# 432 + 4 f words bytes, and each case must land where it says:
#   - mid-base: 2500 words, 8 frames (70,432 bytes: B under 2000, S 0);
#   - mid-SH: 5000 words, 12 frames (220,432 bytes: B 2000, S under 2000);
#   - base-only: 4000 words, 8 frames (112,432 bytes, the base records' end: B 2000, S 0).
# On the web a frame takes what has arrived (the header and more in one read, or less than the
# words), so a run lands wherever it does, and need only be partway (docs/web.md).
# --check-decode poisons the stream-buffer's words past arrived in each frame and fails the run on
# any splat decoded from them, or a decoded count that isn't what the arrived bytes hold.
cat > "$out/degree0.json" <<'EOF'
{"shDegree": 0, "position": {"encoding": "f32"}, "scale": {"encoding": "f32"},
 "rotation": {"encoding": "f32"}, "opacity": {"encoding": "f32"}, "dc": {"encoding": "f32"}, "sh": {"encoding": "f32"}}
EOF
"$python" "$tools/convert.py" stream "$golden/s3/splats.bin" "$out/degree0" --config "$out/degree0.json" \
    > "$out/degree0-convert.log" ||
    { cat "$out/degree0-convert.log"; echo "splat_view: encoding S3 at degree 0 FAILED"; exit 1; }
records() {  # records <log>: the last frame's base and SH records, "B S"
    sed -n 's/^splat_view: streamed .* bytes, \([0-9]*\) base and \([0-9]*\) SH records$/\1 \2/p' "$1"
}
for frames_in_flight in 2 3; do
    for case in "mid-base 2500 8" "mid-sh 5000 12" "base-only 4000 8"; do
        # shellcheck disable=SC2086 # the case's name, words and frames
        set -- $case
        run=stream-$1-$frames_in_flight
        if ! program --validation --frames "$3" --frames-in-flight $frames_in_flight --stream-words "$2" \
            --check-decode --capture "$out/$run.png" "$out/full/scene.stream" \
            --cameras "$golden/s3/cameras.bin" --camera 0 --size 128x128 > "$out/$run.log" 2>&1; then
            fail "$out/$run.log" "$run's run"
            continue
        fi
        # shellcheck disable=SC2046 # two numbers
        set -- "$1" $(records "$out/$run.log")
        if [ -z "${3:-}" ] || ! grep -q '^splat_view: still loading' "$out/$run.log" || ! case $binary:$1 in
            *.js:*) true ;;
            *:mid-base) [ "$2" -gt 0 ] && [ "$2" -lt 2000 ] && [ "$3" = 0 ] ;;
            *:mid-sh) [ "$2" = 2000 ] && [ "$3" -gt 0 ] && [ "$3" -lt 2000 ] ;;
            *) [ "$2" = 2000 ] && [ "$3" = 0 ] ;;
        esac; then
            fail "$out/$run.log" "$run: not partway through as the case needs ('${2:-}' base, '${3:-}' SH)"
            continue
        fi
        "$python" - "$tools" "$golden/s3/splats.bin" "$out/$run-splats.bin" "$2" "$3" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from splatdata.splats import Splats, from_std430, to_std430
with open(sys.argv[2], "rb") as f:
    s = from_std430(f.read())
base, sh = int(sys.argv[4]), int(sys.argv[5])
coefficients = s.sh[:base].copy()
coefficients[sh:, 1:, :] = 0  # the SH above degree 0 of the splats whose SH record hasn't arrived
with open(sys.argv[3], "wb") as f:
    f.write(to_std430(Splats(s.position[:base], s.log_scale[:base], s.rotation[:base],
                             s.opacity_logit[:base], coefficients)))
EOF
        if ! "$python" "$tools/convert.py" stream "$out/$run-splats.bin" "$out/$run-whole" \
            --config "$tools/stream-configs/full.json" > "$out/$run-convert.log"; then
            fail "$out/$run-convert.log" "encoding $run's truncated scene"
            continue
        fi
        if view "$run-whole" "$out/$run-whole/scene.stream" --cameras "$golden/s3/cameras.bin" --camera 0 \
            --size 128x128 && cmp -s "$out/$run.png" "$out/$run-whole.png"; then
            echo "splat_view: $run: $2 base and $3 SH records arrived, the image is that scene's whole"
        else
            cat "$out/$run-whole.log"
            fail "$out/$run.log" "$run against its truncated scene"
            continue
        fi
        if [ "$2" = 2000 ] && [ "$3" = 0 ]; then
            if view "$run-degree0" "$out/degree0/scene.stream" --cameras "$golden/s3/cameras.bin" --camera 0 \
                --size 128x128 && cmp -s "$out/$run.png" "$out/$run-degree0.png"; then
                echo "splat_view: $run: the base without SH is S3 at degree 0"
            else
                cat "$out/$run-degree0.log"
                fail "$out/$run.log" "$run against S3 at degree 0"
            fi
        fi
    done
done
# The mutant that shows --check-decode isn't vacuous: --decode-ahead-mutant (which implies it)
# writes a word of the next, unfinished base record (one that hasn't arrived, so PoisonUnarrived's
# poison) into its splat each frame. The mid-base run must then fail on the check's "written past the records decoded",
# its first failure (not a validation message, a crash or another check). Without the poison the
# word would be zero and go unseen, so this also shows the poison is written.
run=stream-mutant
if program --validation --frames 8 --stream-words 2500 --decode-ahead-mutant \
    --capture "$out/$run.png" "$out/full/scene.stream" --cameras "$golden/s3/cameras.bin" --camera 0 \
    --size 128x128 > "$out/$run.log" 2>&1; then
    fail "$out/$run.log" "$run passed, but must fail:"
elif grep -m 1 FAILED "$out/$run.log" | grep -q 'splats (summed over frames) written past the records decoded'; then
    echo "splat_view: $run: failed on the records decoded ahead, as it must"
else
    fail "$out/$run.log" "$run failed, but not on the records decoded ahead:"
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
    program --validation --frames 40 "$out/full/scene.stream" --cameras "$golden/s3/cameras.bin" --camera 0 --size 128x128 --scripted-moves $moves "$@" \
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
        # And while the stream loads (5000 words a frame: all 472,432 bytes in frame 24), still: each
        # frame bringing bytes projects and sorts, and none after the one that loads it
        log=$out/load-frames.log
        if program --validation --frames 40 --stream-words 5000 --timings "$out/full/scene.stream" \
            --cameras "$golden/s3/cameras.bin" --camera 0 --size 128x128 > "$log" 2>&1; then
            project=$(awk '$1 == "project" { print $NF }' "$log")
            sort=$(awk '$1 == "sortSplats" { print $NF }' "$log")
            # "loaded in frame L; Project ran in P frames, the last frame F"
            # shellcheck disable=SC2046 # three numbers
            set -- $(sed -n 's/^splat_view: loaded in frame \([0-9]*\); Project ran in \([0-9]*\) frames, the last frame \([0-9]*\)$/\1 \2 \3/p' "$log")
            if [ "${1:-}" = 24 ] && [ "$3" = 24 ] && [ "$project" = "$2" ] && [ "$sort" = "$2" ]; then
                echo "splat_view: loading, Project and the sort ran in $2 of 40 frames, the last the one that loaded"
            else
                fail "$log" "loading: Project in '$project' frames and the sort in '$sort', not the app's (loaded in frame '${1:-}', the last projection '${3:-}', not 24):"
            fi
        else
            fail "$log" "the loading run"
        fi
        ;;
esac
exit $status
