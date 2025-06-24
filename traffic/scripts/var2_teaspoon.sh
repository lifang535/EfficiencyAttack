#!/bin/bash

cd ..

# python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0
# python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 0
# python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 0

# python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 2
# python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 2
# python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 2

# python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0 2
# python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 0 2
# python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 0 2

CUDA_VISIBLE_DEVICES=0 bash -c '
python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0
python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 2
python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0 2
' &

CUDA_VISIBLE_DEVICES=1 bash -c '
python pipe_var2.py --model_id 0 --algorithm overload
python pipe_var2.py --model_id 0 --algorithm phantom
python pipe_var2.py --model_id 0 --algorithm slowtrack
' &

CUDA_VISIBLE_DEVICES=2 bash -c '
python pipe_var2.py --model_id 0 --algorithm weighted
python pipe_var2.py --model_id 0 --algorithm unweighted
' &

CUDA_VISIBLE_DEVICES=3 bash -c '
python pipe_var1.py --model_id 0 --algorithm weighted
python pipe_var1.py --model_id 0 --algorithm unweighted
' &

CUDA_VISIBLE_DEVICES=4 bash -c '
python pipe.py --model_id 0 --algorithm weighted
python pipe.py --model_id 0 --algorithm unweighted
' &

wait
echo "job done"

# python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 23
# python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 23
# python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 23

# python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 68
# python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 68
# python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 68

# python pipe_var2.py --model_id 0 --algorithm teaspoon
# python pipe_var2.py --model_id 1 --algorithm teaspoon
# python pipe_var2.py --model_id 2 --algorithm teaspoon