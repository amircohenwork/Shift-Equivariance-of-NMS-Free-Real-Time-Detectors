#!/bin/bash
# Queued behind the cross-check batch. Two jobs, both forced by the review.
#
# 1. THE REPLICATION, ON THE STRONG INSTRUMENTS.
#    The uniform-shift dispersion test failed to replicate the head gap on yolov10n --
#    0/10 curves e2e-worse, and the pre-registered primary curve (x/AP_small) came out
#    non-significant AND opposite in sign. But that instrument was always the weakest of
#    the three: on yolo26n the effect was far larger in the oracle envelope (dAP 3.76 vs
#    3.25) and the instance metrics (dropout 46.5% vs 28.8%). Neither has been run on
#    yolov10n. Concluding "the head effect does not generalise" from the weakest
#    instrument alone would be the same mistake as concluding it does from the strongest.
#
# 2. K=2 ORACLE ON yolo26n.
#    The protocol fixes K=1, justified by "the paper measures <1% further movement" --
#    Manfredi & Wang's detectors, never checked on ours. The committed run shows
#    changed=3641/5000 at iteration 0, so it hit the cap rather than converging.
#    --tag keeps the K=2 artifacts from overwriting the K=1 assignments.
source /tmp/runenv.sh
set -x

# wait for the cross-check batch, but do not wait forever
for _ in $(seq 1 720); do [ -f /tmp/NIGHT_ALL_DONE ] && break; sleep 30; done
[ -f /tmp/NIGHT_ALL_DONE ] || { echo "night batch never finished; running anyway"; }

# ---------------------------------------------------------------- 1. replication
for H in e2e nms; do
  python scripts/03_build_cache.py --model yolov10n --head $H --max-shift 1 --step 1 \
    --device 0 --batch 64 2>&1 | tee -a logs/fu_cache_yolov10n_$H.log
done
touch /tmp/FU_CACHE_DONE

for H in e2e nms; do
  python scripts/04_oracle.py --run-id yolov10n_${H}_m1s1_full_pad114 --device 0 \
    2>&1 | tee -a logs/fu_oracle_yolov10n_$H.log
done
touch /tmp/FU_ORACLE_DONE

# instance metrics (flicker / dropout / class-flip / jitter / Stability Index)
for H in e2e nms; do
  python scripts/05_sweep.py --model yolov10n --head $H --subset 1000 --step 2 \
    --device 0 --batch 64 2>&1 | tee -a logs/fu_sweep_yolov10n_$H.log
done
touch /tmp/FU_SWEEP_DONE

# shift-TTA, so the third yolo26n claim also gets a replication attempt
for H in e2e nms; do
  python scripts/07_shift_tta.py --run-id yolov10n_${H}_m1s1_full_pad114 \
    --oracle-json results/oracle/yolov10n_${H}_m1s1_full_pad114_AP50_all.json \
    2>&1 | tee -a logs/fu_tta_yolov10n_$H.log
done
touch /tmp/FU_TTA_DONE

# ---------------------------------------------------------------- 2. K=2 oracle
for H in e2e nms; do
  python scripts/04_oracle.py --run-id yolo26n_${H}_m1s1_full_pad114 --iters 2 \
    --device 0 2>&1 | tee -a logs/fu_oracleK2_yolo26n_$H.log
done
touch /tmp/FU_K2_DONE

python scripts/06_report.py 2>&1 | tee logs/fu_report.log
touch /tmp/ALL_WORK_DONE
