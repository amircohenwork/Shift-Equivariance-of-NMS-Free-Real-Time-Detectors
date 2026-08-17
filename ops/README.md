# Long-run operations

Scripts for unattended multi-hour batches. Nothing here affects a result — these only
launch and watch the scripts under `scripts/`.

| file | what it does |
|---|---|
| `run_crosscheck.sh` | the cross-model 1-D sweeps + padding ablation, in value order |
| `run_followup.sh` | queued behind it: YOLOv10-N oracle/instance/TTA replication, and the K=2 oracle |
| `watchdog.sh` | liveness monitor; writes `logs/watchdog_status.txt`, raises `/tmp/WATCHDOG_ALERT` |

## Watching a run

```bash
ssh <box> 'cat ~/shifteq/logs/watchdog_status.txt'   # current state
ssh <box> 'tail -5 ~/shifteq/logs/watchdog.jsonl'    # heartbeat history
```

## What the watchdog checks, every 60 s

* the batch runner still exists, and no completion sentinel — i.e. it crashed
* **no progress for 15 min** — measured on shift-rows written, batch log growth, and the
  file count under `results/`
* `Traceback` / `CUDA out of memory` / `CUDA error` in any batch log
* free disk below 8 GB
* `nvidia-smi` answering at all

The 15-minute threshold is set from measurement, not taste: steady-state shifts take 15 s
and the cold-cache first shift took 238 s.

## Test the alert paths before trusting them

A watchdog is a claim about failures you have not seen yet, so the claim needs testing.
Copy it to a sandbox with `ALERT`, the working directory and the sentinel paths rewritten,
then plant each fault: a `Traceback` in a log, a forced stall (`STALL_SECS=1`), an
impossible disk floor (`MIN_FREE_GB=99999`). Check each raises, and — the part that is easy
to skip — check that a **healthy, advancing** run raises nothing.

**The progress signal must exclude the watchdog's own files.** `logs/watchdog_status.txt`
and `logs/watchdog.jsonl` are rewritten every poll. An earlier version summed `du -sb logs
results`, which therefore grew unconditionally: `idle` reset forever and the stall check
could never fire. It reported "alive" through anything. That version passed a casual look
and failed the sandbox test — which is the entire argument for running the test.
