#!/usr/bin/env bash
# COCO val2017 only: images + YOLO labels + the official annotation JSON.
#
# Ultralytics' own coco.yaml `download:` block pulls train2017.zip (19 GB) which this
# project never touches -- everything here is inference on val. It also does NOT fetch
# annotations/instances_val2017.json, which DetectionValidator.eval_json needs for the
# COCO mAP that the Phase-0 gate compares against. Both gaps are handled below.
set -euo pipefail

DATASETS_DIR="${DATASETS_DIR:-$HOME/datasets}"
COCO="$DATASETS_DIR/coco"
mkdir -p "$COCO/images" "$COCO/annotations"
cd "$COCO"

fetch() {  # url, dest
  if [ -f "$2" ]; then echo "have $2"; else echo "-> $1"; curl -fL --retry 3 -o "$2" "$1"; fi
}

echo "== val images =="
fetch "http://images.cocodataset.org/zips/val2017.zip" images/val2017.zip
[ -d images/val2017 ] || unzip -q images/val2017.zip -d images/

echo "== official annotations (needed for COCO mAP) =="
fetch "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" annotations_trainval2017.zip
[ -f annotations/instances_val2017.json ] || unzip -qo annotations_trainval2017.zip \
  "annotations/instances_val2017.json" -d .

echo "== YOLO-format labels (Ultralytics mirror) =="
if [ ! -d labels/val2017 ]; then
  fetch "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip" \
        ../coco2017labels-segments.zip
  unzip -q ../coco2017labels-segments.zip -d "$DATASETS_DIR"
fi

echo "== val2017.txt image list =="
if [ ! -f val2017.txt ]; then
  find "$PWD/images/val2017" -name '*.jpg' | sed "s|$PWD/|./|" | sort > val2017.txt
fi

cat <<EOF

done.
  images : $COCO/images/val2017        ($(ls images/val2017 2>/dev/null | wc -l | tr -d ' ') files)
  labels : $COCO/labels/val2017
  annots : $COCO/annotations/instances_val2017.json
  list   : $COCO/val2017.txt

Point Ultralytics at it once:
  yolo settings datasets_dir="$DATASETS_DIR"
EOF
