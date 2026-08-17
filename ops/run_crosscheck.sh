#!/bin/bash
# The whole overnight batch: cross-checks first, then the padding ablation.
# Every stage drops a sentinel in /tmp so progress is checkable without reading logs,
# and every sweep is per-shift resumable, so a kill costs at most one shift.
source /tmp/runenv.sh
set -x

fft () {  # model head batch subset [pad]
  python scripts/10_sweep1d_fft.py --model "$1" --head "$2" --subset "$4" \
    --pad-value "${5:-114}" --device 0 --batch "$3" \
    2>&1 | tee -a "logs/sweep1d_$1_$2_n$4_pad${5:-114}.log"
}

# ---------------------------------------------------------------- cross-checks
# 1. the replication: the only other released checkpoint carrying both heads
fft yolov10n e2e 64 5000
fft yolov10n nms 64 5000
touch /tmp/CROSS_V10_DONE

# 2. universality on conventional one-to-many detectors
fft yolov8n nms 64 5000
fft yolo11n nms 64 5000
touch /tmp/CROSS_YOLO_DONE

# 3. a different NMS-free design: transformer set prediction, stretch preprocessing
fft rtdetr-l e2e 32 5000
touch /tmp/CROSS_ALL_DONE

python scripts/12_crosscheck.py --subset 5000 2>&1 | tee logs/crosscheck.log
touch /tmp/CROSSCHECK_REPORT_DONE

# ---------------------------------------------------------------- padding ablation
# pad 114 at n1000 already exists from the earlier run; only the other three are needed.
for P in 0 -1 -2; do
  fft yolo26n e2e 64 1000 $P
  fft yolo26n nms 64 1000 $P
done
touch /tmp/PAD_SWEEPS_DONE

python scripts/13_pad_ablation.py --model yolo26n --subset 1000 2>&1 | tee logs/pad_ablation.log
touch /tmp/NIGHT_ALL_DONE
