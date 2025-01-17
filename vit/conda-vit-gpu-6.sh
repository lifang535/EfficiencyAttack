for IDX in {0..90}; do
    export CUDA_VISIBLE_DEVICES=6,7
    echo "Running detr.py on GPU:6 target_cls_idx=$IDX"
    sleep 3
    python detr.py --epoch_num 100 --algo_name overload --target_cls_idx $IDX --val_size 100
    python detr.py --epoch_num 100 --algo_name slow --target_cls_idx $IDX --val_size 100
    python detr.py --epoch_num 100 --algo_name phantom --target_cls_idx $IDX --val_size 100
    python detr.py --epoch_num 100 --algo_name single --target_cls_idx $IDX --val_size 100
done
