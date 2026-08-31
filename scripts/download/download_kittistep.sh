#!/bin/bash
# Usage:  bash scripts/download/download_kittistep.sh   (safe to rerun)
set -euo pipefail
if [[ -z "${KITTISTEP_ROOT:-}" ]]; then eval "$(conda shell.bash hook)"; conda activate video; fi
: "${KITTISTEP_ROOT:?conda activate video}"

mkdir -p "$KITTISTEP_ROOT" && cd "$KITTISTEP_ROOT"

if [[ ! -f .done ]]; then
  # 1. images: KITTI tracking left-camera frames (~15 GB)
  aria2c -x 8 -s 8 -c --auto-file-renaming=false \
    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip
  # 2. labels: STEP panoptic maps (small)
  wget -c https://storage.googleapis.com/gresearch/tf-deeplab/data/kitti-step.tar.gz

  # Note: -o overwrites, so a file truncated by an interrupted unzip is rewritten on rerun (-n would keep it).
  unzip -o -q data_tracking_image_2.zip 'training/*'     # -> training/image_02/<seq>/; testing/ has no labels
  # Note: skip if labels were already moved into place, or a rerun would nest them as panoptic_maps/panoptic_maps.
  [[ -d panoptic_maps ]] || tar xzf kitti-step.tar.gz
  PM=$(find . -maxdepth 3 -type d -name panoptic_maps -not -path './panoptic_maps' | head -1)
  [[ -z "$PM" && -d panoptic_maps ]] && PM=./panoptic_maps
  [[ -n "$PM" ]] || { echo "panoptic_maps not found in kitti-step.tar.gz; check its layout" >&2; exit 1; }

  # 3. split images by the label folders (train/val sequence lists come from the labels)
  for split in train val; do
    mkdir -p images/$split
    for seq in "$PM"/$split/*/; do
      seq=$(basename "$seq")
      [[ -d images/$split/$seq ]] || mv training/image_02/$seq images/$split/
    done
  done
  if [[ "$PM" != ./panoptic_maps ]]; then
    mv "$PM" ./panoptic_maps
    rmdir "$(dirname "$PM")" 2>/dev/null || true
  fi

  # 4. cleanup (test sequences have no labels)
  rm -rf training testing data_tracking_image_2.zip kitti-step.tar.gz
  touch .done
fi

# 5. check: expect 5027 train / 2981 val, images == labels
for s in train val; do
  echo "$s: $(find images/$s -name '*.png' | wc -l) images, $(find panoptic_maps/$s -name '*.png' | wc -l) labels, $(ls images/$s | wc -l) sequences"
done