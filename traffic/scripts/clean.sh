#!/bin/bash

cd ..

# 在 GPU 6 上运行任务
CUDA_VISIBLE_DEVICES=6 bash -c '
python pipe.py --model_id 0  --algorithm clean &&
python pipe.py --model_id 1  --algorithm clean &&
python pipe.py --model_id 2  --algorithm clean
' &

# 在 GPU 7 上运行任务
CUDA_VISIBLE_DEVICES=7 bash -c '
python pipe_var1.py --model_id 0  --algorithm clean &&
python pipe_var1.py --model_id 1  --algorithm clean &&
python pipe_var1.py --model_id 2  --algorithm clean &&

python pipe_var2.py --model_id 0  --algorithm clean &&
python pipe_var2.py --model_id 1  --algorithm clean &&
python pipe_var2.py --model_id 2  --algorithm clean
' &

wait

echo "job done"