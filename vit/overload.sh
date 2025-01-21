
export CUDA_VISIBLE_DEVICES=1
python detr.py --epoch_num 100 --algo_name overload --target_cls_idx 0 --val_size 1000
python rt-detr.py --epoch_num 100 --algo_name overload --target_cls_idx 0 --val_size 1000

