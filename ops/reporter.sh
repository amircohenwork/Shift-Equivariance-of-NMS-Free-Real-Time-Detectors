#!/bin/bash
# Waits for the whole pipeline to finish, then writes one file that answers "what happened".
#
# Why this exists: the local notification channel keeps being killed, so the remote side has
# to leave a complete record rather than assume anyone was watching at the moment it
# finished. Reading logs/SUMMARY.txt afterwards should be as informative as having watched.
cd ~/shifteq || exit 1
OUT=logs/SUMMARY.txt

for _ in $(seq 1 1440); do [ -f /tmp/ALL_WORK_DONE ] && break; sleep 30; done

{
  echo "=============================================================="
  echo " pipeline summary   $(date -Is)"
  echo "=============================================================="
  echo
  echo "--- stages reached ---"
  for s in CROSS_V10_DONE CROSS_YOLO_DONE CROSS_ALL_DONE CROSSCHECK_REPORT_DONE \
           PAD_SWEEPS_DONE NIGHT_ALL_DONE FU_CACHE_DONE FU_ORACLE_DONE FU_SWEEP_DONE \
           FU_TTA_DONE FU_K2_DONE ALL_WORK_DONE; do
    [ -f "/tmp/$s" ] && echo "  reached  $s" || echo "  MISSING  $s"
  done

  echo
  echo "--- errors in any batch log ---"
  grep -lE "Traceback|CUDA out of memory|CUDA error" logs/night.log logs/fu_*.log 2>/dev/null \
    || echo "  none"

  echo
  echo "--- cross-model check (grid-locking universal? head gap replicated?) ---"
  sed "s/\x1b\[[0-9;]*m//g" logs/crosscheck.log 2>/dev/null \
    | grep -E "TOTAL|-> yolo|sign agreement|CROSS-CHECK" || echo "  not produced"

  echo
  echo "--- padding ablation (aliasing or border-reading?) ---"
  sed "s/\x1b\[[0-9;]*m//g" logs/pad_ablation.log 2>/dev/null \
    | grep -E "VERDICT|stride power, constant|across all padding" || echo "  not produced"

  echo
  echo "--- YOLOv10-N oracle: does the head gap replicate on the strong instrument? ---"
  for H in e2e nms; do
    f=results/oracle/yolov10n_${H}_m1s1_full_pad114_AP50_all.json
    [ -f "$f" ] && python3 -c "
import json;d=json.load(open('$f'))
print(f\"  yolov10n/$H  dAP={d['delta']['dAP']*100:.4f}  dAP50={d['delta']['dAP50']*100:.4f}  baseline={d['baseline']['AP']*100:.3f}\")" \
      || echo "  yolov10n/$H  not produced"
  done
  echo "  (yolo26n for comparison: e2e dAP 3.7555, nms dAP 3.2537 -> e2e +15.4%)"

  echo
  echo "--- YOLOv10-N instance metrics ---"
  for H in e2e nms; do
    f=results/sweep/yolov10n_${H}_m31s2_n1000_pad114_summary.json
    [ -f "$f" ] && python3 -c "
import json;d=json.load(open('$f'))['stability']
print(f\"  yolov10n/$H  flicker={d['flicker_rate']*100:.1f}%  dropout={d['dropout_rate']*100:.1f}%  small_dropout={d.get('small_dropout_rate',float('nan'))*100:.1f}%  jitter={d['jitter_px']:.2f}px\")" \
      || echo "  yolov10n/$H  not produced"
  done
  echo "  (yolo26n for comparison: e2e flicker 39.2% dropout 46.5% small 80.2% | nms 30.2% 28.8% 57.7%)"

  echo
  echo "--- K=2 oracle: was K=1 sufficient? ---"
  for H in e2e nms; do
    f=results/oracle/yolo26n_${H}_m1s1_full_pad114_K2_AP50_all.json
    [ -f "$f" ] && python3 -c "
import json;d=json.load(open('$f'))
it=d['best']['iterations']
print(f\"  yolo26n/$H K=2  dAP={d['delta']['dAP']*100:.4f}  iterations={[(i['iter'],i['changed']) for i in it]}\")" \
      || echo "  yolo26n/$H  not produced"
  done
  echo "  (K=1 published: e2e dAP 3.7555, nms dAP 3.2537)"

  echo
  echo "--- last watchdog state ---"
  cat logs/watchdog_status.txt 2>/dev/null
} > "$OUT" 2>&1

touch /tmp/SUMMARY_READY
