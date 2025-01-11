#!/bin/bash

# Iterate from 0 to 90
for IDX in {0..90}; do
    echo "Running script for target_cls_idx=$IDX"
    python detr.py --epoch_num 100 --algo_name ada --pipeline_name caption --target_cls_idx $IDX --val_size 100
done