#!/bin/bash
cd ..

CUDA_VISIBLE_DEVICES=0 bash -c '
python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 0
' &

CUDA_VISIBLE_DEVICES=1 bash -c '
python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 2
' &

CUDA_VISIBLE_DEVICES=2 bash -c '
python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 0 2
' &

CUDA_VISIBLE_DEVICES=5 bash -c '
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 0
' &

CUDA_VISIBLE_DEVICES=6 bash -c '
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 2
' &

CUDA_VISIBLE_DEVICES=7 bash -c '
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 0 2
' &

wait
echo "job done"

# # GPU 0 
# CUDA_VISIBLE_DEVICES=0 bash -c '
# python pipe.py --model_id 0 --algorithm teaspoon --target_idx 0 &
# python pipe.py --model_id 0 --algorithm teaspoon --target_idx 2 &
# python pipe.py --model_id 0 --algorithm teaspoon --target_idx 0 2 &
# wait
# ' &

# # GPU 1
# CUDA_VISIBLE_DEVICES=1 bash -c '
# python pipe.py --model_id 1 --algorithm teaspoon --target_idx 0 &
# python pipe.py --model_id 1 --algorithm teaspoon --target_idx 2 &
# python pipe.py --model_id 1 --algorithm teaspoon --target_idx 0 2 &
# wait
# ' &

# # GPU 2
# CUDA_VISIBLE_DEVICES=2 bash -c '
# python pipe.py --model_id 2 --algorithm teaspoon --target_idx 0 &
# python pipe.py --model_id 2 --algorithm teaspoon --target_idx 2 &
# python pipe.py --model_id 2 --algorithm teaspoon --target_idx 0 2 &
# wait
# ' &

# # GPU 3
# CUDA_VISIBLE_DEVICES=3 bash -c '
# python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 0 &
# python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 2 &
# python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 0 2 &
# wait
# ' &

# # GPU 4
# CUDA_VISIBLE_DEVICES=4 bash -c '
# python pipe_var1.py --model_id 1 --algorithm teaspoon --target_idx 0 &
# python pipe_var1.py --model_id 1 --algorithm teaspoon --target_idx 2 &
# python pipe_var1.py --model_id 1 --algorithm teaspoon --target_idx 0 2 &
# wait
# ' &

# # GPU 5
# CUDA_VISIBLE_DEVICES=5 bash -c '
# python pipe_var1.py --model_id 2 --algorithm teaspoon --target_idx 0 &
# python pipe_var1.py --model_id 2 --algorithm teaspoon --target_idx 2 &
# python pipe_var1.py --model_id 2 --algorithm teaspoon --target_idx 0 2 &
# wait
# ' &

# # GPU 6
# CUDA_VISIBLE_DEVICES=6 bash -c '
# python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0 &
# python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 2 &
# python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0 2 &
# wait
# ' &

# # GPU 7
# CUDA_VISIBLE_DEVICES=7 bash -c '
# python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 0 &
# python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 2 &
# python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 0 2 &
# wait

# python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 0 &
# python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 2 &
# python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 0 2 &
# wait
# ' &

# 等待所有GPU上的任务完成
