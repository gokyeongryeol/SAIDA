#!/bin/bash
set -e

DATA_NAME=$1
SHOT=$2
USE_TTA=$3

python external/mmdetection/tools/test.py \
    /workspace/config/bbox_config.py \
    /workspace/weights/codetr-${DATA_NAME}-${SHOT}.pth \
    --out /workspace/submit/${DATA_NAME}_${SHOT}shot.pkl \
    ${USE_TTA:+--tta}

python utils/pkl_to_json.py \
    --pickle-path /workspace/submit/${DATA_NAME}_${SHOT}shot.pkl \
    --output-json /workspace/submit/${DATA_NAME}_${SHOT}shot.json

rm -f /workspace/submit/${DATA_NAME}_${SHOT}shot.pkl
