for IDX in {33..90}; do
    export CUDA_VISIBLE_DEVICES=6,7

    echo "Running detr.py and rt-detr for single algo target_cls_idx=$IDX"
    sleep 3    
    python detr.py --epoch_num 100 --algo_name single --target_cls_idx $IDX --val_size 100
    python rt-detr.py --epoch_num 100 --algo_name single --target_cls_idx $IDX --val_size 100
done
