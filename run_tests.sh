#!/bin/bash

# A list of prediction thresholds to iterate over
THRESHOLDS="0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# Loop through the thresholds and execute the python script for each
for PRED_THRESHOLD in $THRESHOLDS; do
    echo "Running test with prediction threshold: $PRED_THRESHOLD"
    python test_OPDReal.py \
        --config config/opd_train_large_beta.yaml \
        --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v1/epoch=2-val/loss_total=1.0314.ckpt" \
        --dataset_key MotionNet_test \
        --motion_threshold 10.0 \
        --iou_threshold 0.5 \
        --pred_threshold "$PRED_THRESHOLD"
    echo "-----------------------------------------------------"
done