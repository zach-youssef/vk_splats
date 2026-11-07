#!/bin/sh
# splat_train's regression test, headless with validation: test.sh <binary> <output dir>. It gives
# the committed S3 scene (programs/splat_render/golden/s3) its ground truth with splat_render
# (make_synthetic.py render-gt, the same build as <binary>: native, or the web build for a .js),
# with SH degree 0: 22 training views can't pin down the random degree-3 colours well enough for
# the held-out views in a short run. Two main runs, at 128x128 (and the 160x96 view) with the
# reference's learning rates and densification schedule (from 500, every 100, the screen-space
# prune at the reference's intended 20 px) but densifying until 1500 only, so that each run ends
# with a few hundred plain Adam steps (as the reference's 15000 of 30000: a run ending on a
# densification scores fresh, untrained clones and halves), each checked by
# check.py (the held-out PSNR, the training loss's fall (first epoch to last), nothing NaN,
# the splats in use within --max-splats and the exported PLY, rendered again by splat_render from
# each held-out camera, scoring what splat_train reported):
#   - perturb: the splats perturbed (make_synthetic.py perturb, seed 0), fitted back for 2000
#     iterations, the SH degree rising every 500, with the opacity reset every 900 (so at 900,
#     and the size prunes after 900). Densification fills --max-splats 4096 by iteration 800, so
#     later additions are dropped (the overflow). It passes at a held-out PSNR within 0.5 dB of
#     the native builds' (native_perturb_psnr: 35.94 on lavapipe, 35.91 on MoltenVK; the web's
#     35.89 on lavapipe, 35.86 on Metal; the start is 20), so it is also the web's parity check
#     (docs/web.md#training-on-the-web); a native build also fails more than 1 dB above it, so
#     the reference can't go stale by more. It also needs a loss falling 8 times (34.0). The tile
#     entries' capacity starts at 256, so the first iterations overflow and grow it: check.py
#     requires skipped iterations (8 here), which covers Adam's skip, Record's -1 and their report;
#   - sparse: initialized as the converter initializes COLMAP points from a tenth of the splats'
#     positions (make_synthetic.py sparse, seed 0: 200 splats), trained for 1800 iterations with
#     --max-splats 12288; densification must grow the splats (it reached 11470 of the 12288 on lavapipe) and the
#     held-out PSNR reach 28 dB (37.0 on MoltenVK; a run ending on a reset scores about 13). Its
#     PSNR swings by up to 0.7 dB with the least change to its inputs (docs/web.md#training-on-the-web),
#     so one run can't hold it within 0.5 dB of native, and it isn't held to that.
#   - last: from perturb's start, 300 iterations densifying from 100 with --reset-interval 300, so
#     iteration 200 densifies and the last, 300, would densify and reset if it weren't the last:
#     the reference scores and saves an iteration before its densification and reset. It must
#     densify once and score 22 dB (26.7 on MoltenVK, lavapipe and the web, the loss falling 3.3
#     times; a reset on the last iteration scored 13.3);
#   - chunks: the perturb start trained for 300 iterations (densifying at 100 and 200, the opacity
#     reset at 200) twice, the backward pass in one chunk and in chunks of 997 entries (frames
#     need up to about 12000: a dozen chunks, of 41 the graph holds at --max-entries 40000); the
#     PLYs must be the same, byte for byte. The capacity starts above what any frame needs: when a
#     growth lands depends on the counts' readback, so runs that grow aren't the same bit for bit;
#   - max-entries: a run whose frames need more than --max-entries 1000 must fail, saying so.
# perturb and sparse train on (1 - 0.2) L1 + 0.2 D-SSIM (--lambda-dssim's default); on L1 alone they scored 35.9
# and 32.9 dB, the loss falling 16.3 and 8.8 times.
# Each run prints its PSNR and splat count, so a CI log shows the margins.
# $PYTHON (python3 by default) needs numpy and Pillow.
set -eu
binary=$1
out=$2
python=${PYTHON:-python3}
here=$(dirname "$0")
# splat_render's build beside this one: build/splat_train/splat_train or
# build/web/splat_train/splat_train.zgal.main.js
renderer=$(printf '%s\n' "$binary" | sed 's/splat_train/splat_render/g')

rm -rf "$out"
mkdir -p "$out"

program() {  # the native binary, or the web build under Node
    case $binary in
        *.js) "${NODE:-node}" "$binary" "$@" ;;
        *) "$binary" "$@" ;;
    esac
}

fail() {
    [ -n "${log:-}" ] && [ -f "$log" ] && cat "$log"
    echo "splat_train: FAILED, $1"
    exit 1
}

cp -R programs/splat_render/golden/s3 "$out/gt"
"$python" tools/make_synthetic.py render-gt "$out/gt" "$renderer" --validation --sh-degree 0 ||
    fail "rendering the ground truth"

train() {  # train <name> <make_synthetic.py command> <iterations> <check.py arguments> -- <splat_train arguments>
    name=$1
    command=$2
    iterations=$3
    shift 3
    dir="$out/$name"
    log="$dir/train.log"
    mkdir -p "$dir"
    ln -s ../gt "$dir/gt"
    "$python" tools/make_synthetic.py "$command" "$out/gt" "$dir/start" --seed 0 > /dev/null ||
        fail "making the $name start"
    checks=
    while [ "$1" != -- ]; do checks="$checks $1"; shift; done
    shift
    # The loading frames, the iterations, the held-out views and the readback's latency
    # shellcheck disable=SC2046 # the arguments are words without spaces
    SPLAT_TRAIN_PLY="$dir/trained.ply" program --validation --frames $((iterations + 64)) \
        $("$python" tools/scene_args.py --train "$dir/start") --iterations "$iterations" --sh-interval 500 \
        "$@" > "$log" 2>&1 || fail "the $name run exited with $?"
    grep -e 'held-out PSNR' -e 'splats in use' "$log"
    # shellcheck disable=SC2086 # the check's arguments are words without spaces
    "$python" "$here/check.py" "$dir" "$renderer" $checks || fail "the $name check"
}

# perturb's held-out PSNR on the native builds; every build must come within 0.5 dB of it. Update
# it (from CI's native log) when a change to training moves it: a native build fails more than
# 1 dB above it (test.sh's seed scores within 0.05 dB of it on lavapipe and MoltenVK), so the
# reference, and the web's bound with it, can't fall more than 1 dB behind native. The web isn't
# bounded above.
native_perturb_psnr=35.9
offset() { LC_ALL=C awk -v p="$native_perturb_psnr" -v d="$1" 'BEGIN { printf "%.2f", p + d }'; }
case $binary in
    *.js) ceiling= ;;
    *) ceiling="--max-psnr $(offset 1.0)" ;;
esac
# shellcheck disable=SC2086 # the ceiling is no word or two, split on purpose
train perturb perturb 2000 --min-psnr "$(offset -0.5)" $ceiling --min-ratio 8 --min-skipped 1 -- \
    --initial-entries 256 --max-splats 4096 --max-screen-size 20 --reset-interval 900 \
    --densify-until 1500
train sparse sparse 1800 --min-psnr 28 --min-ratio 5 --min-skipped 0 --grow -- \
    --max-splats 12288 --max-screen-size 20 --densify-until 1500 --sh-degree 0
train last perturb 300 --min-psnr 22 --min-ratio 2 --min-skipped 0 --densifications 1 -- \
    --max-splats 4096 --max-screen-size 20 --densify-from 100 --reset-interval 300

# The backward pass in chunks trains the same splats, bit for bit, as in one
dir="$out/chunks"
mkdir -p "$dir"
"$python" tools/make_synthetic.py perturb "$out/gt" "$dir/start" --seed 0 > /dev/null || fail "making the chunks start"
for chunks in one many; do
    log="$dir/$chunks.log"
    case $chunks in
        one) extra= ;;
        *) extra="--max-entries 40000 --chunk-entries 997" ;;
    esac
    # shellcheck disable=SC2046,SC2086 # the arguments are words without spaces
    SPLAT_TRAIN_PLY="$dir/$chunks.ply" program --validation --frames 364 \
        $("$python" tools/scene_args.py --train "$dir/start") --iterations 300 --sh-interval 100 \
        --initial-entries 20000 --max-splats 4096 --max-screen-size 20 --densify-from 100 \
        --densify-interval 100 --reset-interval 200 --densify-until 300 $extra > "$log" 2>&1 ||
        fail "the chunks run ($chunks) exited with $?"
done
log=
cmp -s "$dir/one.ply" "$dir/many.ply" || fail "training in chunks of 997 entries differs from one chunk"
echo "splat_train: chunks of 997 entries train the same splats as one chunk"

# A frame needing more than --max-entries fails the run, naming the cap
dir="$out/max-entries"
mkdir -p "$dir"
log="$dir/train.log"
# shellcheck disable=SC2046 # the arguments are words without spaces
if program --validation --frames 84 $("$python" tools/scene_args.py --train "$out/chunks/start") \
        --iterations 20 --initial-entries 256 --max-entries 1000 --max-splats 4096 --max-screen-size 20 > "$log" 2>&1 ||
        ! grep -q "tile entries, more than --max-entries 1000" "$log"; then
    fail "a frame past --max-entries didn't fail the run"
fi
echo "splat_train: a frame past --max-entries fails the run"
