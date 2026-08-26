#!/bin/bash
# Usage:  bash scripts/download_cityscapes.sh [--splits val] [--workers 32]
set -euo pipefail
if [[ -z "${CITYSCAPES_ROOT:-}" ]]; then eval "$(conda shell.bash hook)"; conda activate video; fi
: "${CITYSCAPES_ROOT:?conda activate video}"

python "$(dirname "$0")/download_cityscapes.py" "$@"

# cleanup: zips, partials, temp files
cd "$CITYSCAPES_ROOT"
rm -f -- *.zip *.part
find . -name '*.tmp' -type f -delete

echo "train frames: $(find leftImg8bit_sequence/train -name '*.png' 2>/dev/null | wc -l) / 89250"
echo "val frames:   $(find leftImg8bit_sequence/val   -name '*.png' 2>/dev/null | wc -l) / 15000"
echo "train labels: $(find gtFine/train -name '*_labelTrainIds.png' 2>/dev/null | wc -l) / 2975"
echo "val labels:   $(find gtFine/val   -name '*_labelTrainIds.png' 2>/dev/null | wc -l) / 500"
