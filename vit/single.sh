for IDX in {0..90}; do
    export CUDA_VISIBLE_DEVICES=5
    echo "Running script for target_cls_idx=$IDX"
    python detr.py --epoch_num 100 --algo_name single --target_cls_idx $IDX --val_size 100
    python rt-detr.py --epoch_num 100 --algo_name single --target_cls_idx $IDX --val_size 100
done
