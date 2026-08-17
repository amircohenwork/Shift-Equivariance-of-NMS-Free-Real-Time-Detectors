MODEL ?= yolo26n
HEAD ?= e2e
DEVICE ?= cuda
SUBSET ?= 1000
BATCH ?= 64
STEP ?= 2

.PHONY: help setup test phase0 phase0-cross cache-oracle oracle-budget oracle sweep sweep-dense report fft crosscheck-sweeps crosscheck cache-benefit m-ladder tta pad-ablation pad-ablation-report clean-results

# arms of the cross-check: yolov10n is the second dual-head checkpoint (the replication),
# the rest test whether grid-locking is a property of detectors rather than of YOLO26.
CROSS_MODELS ?= yolov8n yolo11n yolov10n rtdetr-l
CROSS_ARMS ?= yolo26n:e2e yolo26n:nms yolov10n:e2e yolov10n:nms yolov8n:nms yolo11n:nms rtdetr-l:e2e
PAD_VALUES ?= 0 -1 -2

help:
	@grep -E '^[a-z0-9-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

setup:            ## download COCO val2017 + annotations
	bash scripts/00_setup_coco.sh

test:             ## unit tests (no GPU, no dataset needed)
	python -m pytest tests -q

phase0:           ## GATE A + GATE B -- run this before anything else
	python scripts/01_reproduce_published.py --model $(MODEL) --device $(DEVICE) --batch $(BATCH)
	python scripts/02_harness_gates.py --model $(MODEL) --head e2e --device $(DEVICE) --batch $(BATCH)
	python scripts/02_harness_gates.py --model $(MODEL) --head nms --device $(DEVICE) --batch $(BATCH)

cache-oracle:     ## inference for the M=1 oracle grid, both heads
	python scripts/03_build_cache.py --model $(MODEL) --head e2e --max-shift 1 --step 1 \
		--device $(DEVICE) --batch $(BATCH)
	python scripts/03_build_cache.py --model $(MODEL) --head nms --max-shift 1 --step 1 \
		--device $(DEVICE) --batch $(BATCH)

oracle-budget:    ## time one AP call and print the K x (M+1)^2 x N budget before committing
	python scripts/04_oracle.py --run-id $(MODEL)_e2e_m1s1_full_pad114 --dry-run --device $(DEVICE)

oracle:           ## greedy AP_best / AP_worst, both heads
	python scripts/04_oracle.py --run-id $(MODEL)_e2e_m1s1_full_pad114 --device $(DEVICE)
	python scripts/04_oracle.py --run-id $(MODEL)_nms_m1s1_full_pad114 --device $(DEVICE)

sweep:            ## uniform-shift curve AP(delta), both heads
	python scripts/05_sweep.py --model $(MODEL) --head e2e --subset $(SUBSET) --step $(STEP) \
		--device $(DEVICE) --batch $(BATCH)
	python scripts/05_sweep.py --model $(MODEL) --head nms --subset $(SUBSET) --step $(STEP) \
		--device $(DEVICE) --batch $(BATCH)

sweep-dense:      ## step-1 control run that validates the step-2 shortcut
	python scripts/05_sweep.py --model $(MODEL) --head e2e --subset 100 --step 1 --tag dense \
		--device $(DEVICE) --batch $(BATCH)

report:           ## tables + figures
	python scripts/06_report.py

clean-results:
	rm -rf results

m-ladder:         ## replicate Manfredi & Wang Tab. 2: max-shift ladder on 100 images
	@for M in 1 3 7 15; do \
	  python scripts/03_build_cache.py --model $(MODEL) --head $(HEAD) --max-shift $$M --step 1 \
	    --subset 100 --device $(DEVICE) --batch $(BATCH); \
	  python scripts/04_oracle.py --run-id $(MODEL)_$(HEAD)_m$${M}s1_n100_pad114 \
	    --device $(DEVICE) --no-size-summary; \
	done

tta:              ## shift test-time augmentation: nms vs topk vs vote (needs cache-oracle)
	python scripts/07_shift_tta.py --run-id $(MODEL)_e2e_m1s1_full_pad114 \
		--oracle-json results/oracle/$(MODEL)_e2e_m1s1_full_pad114_AP50_all.json
	python scripts/07_shift_tta.py --run-id $(MODEL)_nms_m1s1_full_pad114 \
		--oracle-json results/oracle/$(MODEL)_nms_m1s1_full_pad114_AP50_all.json

pad-ablation:     ## padding ablation through the spectral instrument (needs `make fft` for pad114)
	@for P in $(PAD_VALUES); do \
	  for H in e2e nms; do \
	    python scripts/10_sweep1d_fft.py --model $(MODEL) --head $$H --subset $(SUBSET) \
	      --pad-value $$P --device $(DEVICE) --batch $(BATCH); \
	  done; \
	done
	$(MAKE) pad-ablation-report

pad-ablation-report:  ## does removing the constant-padding position cue remove the grid-locking?
	python scripts/13_pad_ablation.py --model $(MODEL) --subset $(SUBSET)

fft:              ## 1-D sweep at 1-px resolution + FFT grid-locking test + head-gap gate
	python scripts/10_sweep1d_fft.py --model $(MODEL) --head e2e --subset $(SUBSET) \
		--device $(DEVICE) --batch $(BATCH)
	python scripts/10_sweep1d_fft.py --model $(MODEL) --head nms --subset $(SUBSET) \
		--device $(DEVICE) --batch $(BATCH)
	python scripts/11_head_gap.py --model $(MODEL) --subset $(SUBSET)

cache-benefit:    ## measure what the preprocessing cache is worth on this machine
	python scripts/09_cache_benefit.py --device $(DEVICE) --batch $(BATCH)

phase0-cross:     ## Gate A for every cross-check model -- do this before quoting any of them
	@for M in $(CROSS_MODELS); do \
	  python scripts/01_reproduce_published.py --model $$M --device $(DEVICE) --batch $(BATCH); \
	done

crosscheck-sweeps:  ## 1-D sweep + FFT for every cross-check arm on the full val set (~5.5 h on an A10)
	python scripts/10_sweep1d_fft.py --model yolov10n --head e2e --subset 5000 --device $(DEVICE) --batch 64
	python scripts/10_sweep1d_fft.py --model yolov10n --head nms --subset 5000 --device $(DEVICE) --batch 64
	python scripts/10_sweep1d_fft.py --model yolov8n  --head nms --subset 5000 --device $(DEVICE) --batch 64
	python scripts/10_sweep1d_fft.py --model yolo11n  --head nms --subset 5000 --device $(DEVICE) --batch 64
	python scripts/10_sweep1d_fft.py --model rtdetr-l --head e2e --subset 5000 --device $(DEVICE) --batch 32

crosscheck:       ## universality of grid-locking + does the head gap replicate on yolov10n?
	python scripts/12_crosscheck.py --arms $(CROSS_ARMS) --subset 5000
