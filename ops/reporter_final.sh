#!/bin/bash
# Waits for run_final.sh, then leaves one file that answers "what happened" without anyone
# having watched. Same reason as reporter.sh: the local notification channel is unreliable,
# so the record has to live on the box.
#
# Every number is printed beside the prediction PROTOCOL.md §12 recorded before the run, so
# a reader can see whether it was met without going and looking the criterion up -- and so
# that a result which went against us is as visible as one that did not.
cd ~/shifteq || exit 1
OUT=logs/SUMMARY_FINAL.txt

for _ in $(seq 1 1440); do [ -f /tmp/ALL_WORK_DONE ] && break; sleep 30; done

{
  echo "=============================================================="
  echo " final batch summary   $(date -Is)"
  echo " protocol v1.2.0 §12   (small-object oracle + max-shift ladder)"
  echo "=============================================================="

  echo
  echo "--- stages reached ---"
  for s in FINAL_PREFLIGHT_DONE FINAL_SMALL_DONE FINAL_LADDER_e2e_DONE \
           FINAL_LADDER_nms_DONE FINAL_LADDER_DONE ALL_WORK_DONE; do
    [ -f "/tmp/$s" ] && echo "  reached  $s" || echo "  MISSING  $s"
  done

  echo
  echo "--- errors in any batch log ---"
  grep -lE "Traceback|CUDA out of memory|CUDA error" logs/final_*.log 2>/dev/null || echo "  none"

  echo
  echo "--- §12 verdicts, as the script read them against the pre-registration ---"
  sed "s/\x1b\[[0-9;]*m//g" logs/final_verdicts.log 2>/dev/null \
    | grep -E "===|prediction:|small:|dAP50 ratio|-> |M= " || echo "  not produced"

  echo
  echo "--- §12.1 raw: small-object oracle vs the committed all-area oracle ---"
  python3 - <<'PY' 2>/dev/null || echo "  not produced"
import json, pathlib
d = pathlib.Path("results/oracle")
for m in ("yolo26n", "yolov10n"):
    for h in ("e2e", "nms"):
        rid = f"{m}_{h}_m1s1_full_pad114"
        s, a = d / f"{rid}_small_AP50_small.json", d / f"{rid}_AP50_all.json"
        if not s.exists():
            print(f"  {m}/{h:3s}  small: not produced"); continue
        S, A = json.load(open(s)), json.load(open(a))
        print(f"  {m}/{h:3s}  small dAP={S['delta']['dAP']*100:7.4f}  dAP50={S['delta']['dAP50']*100:7.4f}"
              f"   | all-area dAP={A['delta']['dAP']*100:7.4f}  dAP50={A['delta']['dAP50']*100:7.4f}")
PY
  echo "  (published all-area gaps: yolo26n e2e/nms +15.4%, yolov10n +26.8%)"
  echo "  (prediction: the SMALL gap should be LARGER than these)"

  echo
  echo "--- §12.2 raw: the ladder, read as a shape ---"
  python3 - <<'PY' 2>/dev/null || echo "  not produced"
import json, pathlib
d = pathlib.Path("results/oracle")
for h in ("e2e", "nms"):
    for M in (1, 3, 7, 15):
        p = d / f"yolo26n_{h}_m{M}s1_n100_pad114_AP50_all.json"
        if not p.exists():
            print(f"  yolo26n/{h:3s} M={M:2d}  not produced"); continue
        r = json.load(open(p)); b50 = r["baseline"]["AP50"] * 100
        print(f"  yolo26n/{h:3s} M={M:2d}  {len(r['shifts']):3d} shifts  "
              f"dAP={r['delta']['dAP']*100:6.3f}  dAP50={r['delta']['dAP50']*100:6.3f}  "
              f"best-base={r['delta']['AP50_best']*100-b50:+6.3f}  "
              f"base-worst={b50-r['delta']['AP50_worst']*100:+6.3f}")
PY
  echo "  (Manfredi & Wang RetinaNet, 100 imgs: baseline AP 43.8 / AP50 64.2;"
  echo "   at M=15, dAP 8.3 and dAP50 21.5. Shape only -- different detector, different images.)"
  echo "  (predictions: dAP50 monotone AND sub-linear in M; the two envelope arms roughly equal)"

  echo
  echo "--- last watchdog state ---"
  cat logs/watchdog_status.txt 2>/dev/null
} > "$OUT" 2>&1

touch /tmp/SUMMARY_READY
