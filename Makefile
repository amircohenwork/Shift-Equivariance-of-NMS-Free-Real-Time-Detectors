MODEL ?= yolo26n
HEAD ?= e2e
DEVICE ?= cuda
SUBSET ?= 1000
BATCH ?= 64
STEP ?= 2

.PHONY: help setup test phase0 cache-oracle oracle-budget oracle sweep sweep-dense report m-ladder tta pad-ablation clean-results

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

pad-ablation:     ## padding-value ablation (Kayhan & van Gemert: padding is a position cue)
	@for P in 114 0 -1; do \
	  python scripts/05_sweep.py --model $(MODEL) --head $(HEAD) --subset $(SUBSET) --step $(STEP) \
	    --pad-value $$P --device $(DEVICE) --batch $(BATCH); \
	done
