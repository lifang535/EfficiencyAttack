#!/bin/bash
cd ..

# GPU 0 
CUDA_VISIBLE_DEVICES=3 bash -c '
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 0 --defense --ps_path ./defense &
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 2 --defense --ps_path ./defense &
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 0 2 --defense --ps_path ./defense &
wait
' &

CUDA_VISIBLE_DEVICES=4 bash -c '
python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 0 --defense --ps_path ./defense_var1 &
python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 2 --defense --ps_path ./defense_var1 &
python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 0 2 --defense --ps_path ./defense_var1 &
wait
' &

CUDA_VISIBLE_DEVICES=5 bash -c '
python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0 --defense --ps_path ./defense_var2 &
python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 2 --defense --ps_path ./defense_var2 &
python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0 2 --defense --ps_path ./defense_var2 &
wait
' &