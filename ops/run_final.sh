#!/bin/bash
# The last two experiments in the proposal. Both were declared as intended variations in
# PROTOCOL.md section 9 from v1.0.0; v1.2.0 section 12 states what each predicts and what
# would refute it, written before this script was run.
#
# 1. SMALL-OBJECT ORACLE (section 12.1).
#    Section 4 pins the greedy objective's area range to `all`. The AP_small figure sitting
#    next to it comes from `summary_at_assignment` -- AP_small read off the assignment that
#    maximised OVERALL AP50 -- which describes that assignment rather than searching over
#    small objects, and the README caveat says so. Finding 1's strongest sub-claim is about
#    small objects (80.2% vs 57.7% dropout), so it deserves a real oracle rather than a
#    by-product. Reuses the committed caches: no new inference.
#
#    --tag small is not optional. Without it the run overwrites
#    <run_id>_{best,worst}_assignment.npy, which are the evidence for the published dAP.
#
# 2. MAX-SHIFT LADDER (section 12.2).
#    Manfredi & Wang Table 2: M in {1,3,7,15} on 100 images. The only place we meet the
#    prior work on its own protocol rather than on ours. Read as a SHAPE -- monotone,
#    sub-linear, symmetric about the baseline -- never as levels: its M=1 point is N=100
#    and is not the published M=1/N=5000 number.
set -u
cd "$(dirname "$0")/.." || exit 1

# /tmp is cleared by a reboot, so do not depend on a runenv.sh left by an earlier session.
# The venv lives at ~/venv on this box, not inside the repo; check both rather than assume.
if [ -f /tmp/runenv.sh ]; then source /tmp/runenv.sh; fi
if [ -z "${VIRTUAL_ENV:-}" ]; then
  for V in venv "$HOME/venv" .venv "$HOME/.venv"; do
    [ -f "$V/bin/activate" ] && { source "$V/bin/activate"; break; }
  done
fi
python -c 'import torch' 2>/dev/null || { echo "FATAL: no python with torch on PATH"; exit 1; }
mkdir -p logs results/oracle

DEV=${DEV:-0}
BATCH=${BATCH:-64}
set -x

# ---------------------------------------------------------------- preflight
# The small oracle consumes caches built in an earlier session. A deallocated VM keeps its
# disk, so they should be here -- but "should" is not a check, and discovering it three
# hours in is worse than discovering it now.
MISSING=""
for ARM in yolo26n_e2e yolo26n_nms yolov10n_e2e yolov10n_nms; do
  [ -f "cache/dets/${ARM}_m1s1_full_pad114/manifest.json" ] || MISSING="$MISSING $ARM"
done
if [ -n "$MISSING" ]; then
  echo "REBUILDING missing m1s1 caches:$MISSING"
  for ARM in $MISSING; do
    M=${ARM%_*}; H=${ARM##*_}
    python scripts/03_build_cache.py --model "$M" --head "$H" --max-shift 1 --step 1 \
      --device "$DEV" --batch "$BATCH" 2>&1 | tee -a "logs/final_recache_${ARM}.log"
  done
fi
touch /tmp/FINAL_PREFLIGHT_DONE

# ---------------------------------------------------------------- 1. small-object oracle
# yolo26n first: it is the primary arm, and if the box dies afterwards the claim the README
# actually needs is already closed. yolov10n is the replication of the same prediction.
for ARM in yolo26n_e2e yolo26n_nms yolov10n_e2e yolov10n_nms; do
  python scripts/04_oracle.py --run-id "${ARM}_m1s1_full_pad114" --area small --tag small \
    --device "$DEV" 2>&1 | tee -a "logs/final_smalloracle_${ARM}.log"
done
touch /tmp/FINAL_SMALL_DONE

# ---------------------------------------------------------------- 2. max-shift ladder
# Ascending in M so the cheap rungs land early; M=15 is 256 shifts and dominates the cost.
for H in e2e nms; do
  for M in 1 3 7 15; do
    python scripts/03_build_cache.py --model yolo26n --head "$H" --max-shift "$M" --step 1 \
      --subset 100 --device "$DEV" --batch "$BATCH" 2>&1 | tee -a "logs/final_ladder_${H}.log"
    python scripts/04_oracle.py --run-id "yolo26n_${H}_m${M}s1_n100_pad114" \
      --device "$DEV" --no-size-summary 2>&1 | tee -a "logs/final_ladder_${H}.log"
  done
  touch "/tmp/FINAL_LADDER_${H}_DONE"
done
touch /tmp/FINAL_LADDER_DONE

# ---------------------------------------------------------------- 3. read against §12
# 14 states each result beside the prediction PROTOCOL.md §12 recorded before the run,
# including when the prediction loses. 06 refreshes the standing tables.
python scripts/14_final_report.py 2>&1 | tee logs/final_verdicts.log
python scripts/06_report.py 2>&1 | tee logs/final_report.log
touch /tmp/ALL_WORK_DONE
