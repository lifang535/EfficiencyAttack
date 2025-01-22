#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Iterate from 0 to 90
for IDX in {1,25,78}; do
    echo "Running script for target_cls_idx=$IDX"
    python detr.py --epoch_num 200 --algo_name ada --target_cls_idx $IDX --val_size 1000
done