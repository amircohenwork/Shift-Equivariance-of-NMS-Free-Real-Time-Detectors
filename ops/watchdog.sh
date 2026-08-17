#!/bin/bash
# Watchdog for the overnight batch.
#
# A sentinel file only tells you the run *finished*. It cannot distinguish "still working"
# from "died three hours ago" from "wedged on a stuck CUDA call" -- and all three look
# identical to something that is just waiting for the sentinel to appear. This polls the
# things that actually go wrong and raises /tmp/WATCHDOG_ALERT with a reason, so the failure
# is what gets noticed rather than the silence.
#
# Checks, every POLL seconds:
#   1. the batch process still exists (and if not, whether it exited cleanly)
#   2. progress is advancing -- shift rows written, or the log growing
#   3. no traceback / CUDA OOM in the log
#   4. disk headroom (RT-DETR needs a fresh 6.1 GB stretch cache, padding needs ~3.7 GB more)
#   5. the GPU is still visible to the driver
#
# Writes:
#   logs/watchdog.jsonl        append-only history, one record per poll
#   logs/watchdog_status.txt   current one-screen summary (this is the thing to read)
#   /tmp/WATCHDOG_ALERT        created once, with the reason, when something is wrong

cd ~/shifteq || exit 1
POLL=${POLL:-60}
STALL_SECS=${STALL_SECS:-900}   # 15 min. RT-DETR shifts take ~70 s and a cold-cache first
                                # shift took 238 s, so this is a safe multiple of both.
MIN_FREE_GB=${MIN_FREE_GB:-8}

STATUS=logs/watchdog_status.txt
HIST=logs/watchdog.jsonl
ALERT=/tmp/WATCHDOG_ALERT
rm -f "$ALERT"

last_progress=$(date +%s)
last_sig=""

alert () {  # reason
  [ -f "$ALERT" ] && return
  echo "$(date -Is) $1" > "$ALERT"
  echo "ALERT: $1" >> "$STATUS"
}

while true; do
  now=$(date +%s)
  ts=$(date +%H:%M:%S)

  # --- what stage are we on, and how far in? -----------------------------------------
  cmd=$(ps -eo cmd | grep -E "scripts/(0[3457]|1[03])_" | grep -v grep | head -1)
  alive=$([ -n "$cmd" ] && echo 1 || echo 0)
  stage=$(echo "$cmd" | sed -E 's/.*--model ([^ ]+) --head ([^ ]+).*--pad-value ([^ ]+).*/\1:\2 pad\3/')
  [ -z "$stage" ] && stage="(none)"

  newest=$(ls -t results/sweep1d/*.jsonl 2>/dev/null | head -1)
  rows=$([ -n "$newest" ] && wc -l < "$newest" || echo 0)
  # newest mtime + total size across everything either batch writes. The follow-up's
  # oracle emits no sweep rows and logs to its own file, so keying the stall detector on
  # night.log alone would raise a false alert for 45 minutes of healthy greedy search.
  # MUST exclude the watchdog's own output. logs/watchdog_status.txt and
  # logs/watchdog.jsonl are rewritten every poll, so any progress signal that includes
  # them grows unconditionally: `idle` resets forever and the stall check can never fire.
  # That is a watchdog which reports "alive" no matter what has happened -- strictly worse
  # than no watchdog, because it is trusted. Sum only the batch runners' logs, plus a file
  # count under results/.
  logsz=$(ls -l logs/*.log 2>/dev/null | grep -v watchdog | awk '{s+=$5} END {print s+0}')
  nres=$(find results -type f 2>/dev/null | wc -l)
  sig="$newest:$rows:$logsz:$nres"

  if [ "$sig" != "$last_sig" ]; then
    last_progress=$now
    last_sig="$sig"
  fi
  idle=$((now - last_progress))

  # --- sentinels reached so far --------------------------------------------------------
  done_stages=""
  for s in CROSS_V10_DONE CROSS_YOLO_DONE CROSS_ALL_DONE CROSSCHECK_REPORT_DONE \
           PAD_SWEEPS_DONE NIGHT_ALL_DONE FU_CACHE_DONE FU_ORACLE_DONE FU_SWEEP_DONE FU_TTA_DONE FU_K2_DONE ALL_WORK_DONE; do
    [ -f "/tmp/$s" ] && done_stages="$done_stages $s"
  done

  free_gb=$(df --output=avail -BG ~ | tail -1 | tr -dc '0-9')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
        || echo "NO GPU")

  # --- the actual checks ---------------------------------------------------------------
  if [ "$alive" = "0" ] && [ ! -f /tmp/ALL_WORK_DONE ]; then
    # no worker process and no completion sentinel: either between stages, or dead.
    if ! pgrep -f "run_night.sh|run_followup.sh" > /dev/null; then
      alert "no batch runner and no ALL_WORK_DONE -- crashed or was killed"
    fi
  fi
  if [ "$idle" -gt "$STALL_SECS" ] && [ ! -f /tmp/ALL_WORK_DONE ]; then
    alert "no progress for ${idle}s (stage $stage, $rows rows in $(basename "${newest:-none}"))"
  fi
  errfile=$(grep -lE "Traceback|CUDA out of memory|CUDA error" logs/night.log logs/fu_*.log 2>/dev/null | head -1)
  if [ -n "$errfile" ]; then
    alert "error text in $errfile: $(grep -oE 'Traceback|CUDA out of memory|CUDA error' "$errfile" | tail -1)"
  fi
  if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
    alert "only ${free_gb}G free on / -- RT-DETR still needs a 6.1 GB stretch cache"
  fi
  [ "$gpu" = "NO GPU" ] && alert "nvidia-smi does not answer -- driver or VM problem"

  # --- report --------------------------------------------------------------------------
  {
    echo "watchdog $ts   (poll ${POLL}s, stall threshold ${STALL_SECS}s)"
    echo "  stage      : $stage   [process $([ "$alive" = 1 ] && echo alive || echo absent)]"
    echo "  progress   : $rows rows in $(basename "${newest:-none}"), idle ${idle}s"
    echo "  finished   :${done_stages:- none yet}"
    echo "  disk / gpu : ${free_gb}G free | $gpu"
    [ -f "$ALERT" ] && echo "  ALERT      : $(cat "$ALERT")"
    [ -f /tmp/ALL_WORK_DONE ] && echo "  ALL DONE"
  } > "$STATUS"

  printf '{"t":"%s","stage":"%s","alive":%s,"rows":%s,"idle":%s,"free_gb":%s,"gpu":"%s","alert":%s}\n' \
    "$(date -Is)" "$stage" "$alive" "$rows" "$idle" "${free_gb:-0}" "$gpu" \
    "$([ -f "$ALERT" ] && echo 1 || echo 0)" >> "$HIST"

  [ -f /tmp/ALL_WORK_DONE ] && break
  sleep "$POLL"
done
