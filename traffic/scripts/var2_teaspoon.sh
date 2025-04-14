#!/bin/bash

cd ..

python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0
python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 0
python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 0

python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 2
python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 2
python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 2

python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0 2
python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 0 2
python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 0 2

# python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 23
# python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 23
# python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 23

# python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 68
# python pipe_var2.py --model_id 1 --algorithm teaspoon --target_idx 68
# python pipe_var2.py --model_id 2 --algorithm teaspoon --target_idx 68

# python pipe_var2.py --model_id 0 --algorithm teaspoon
# python pipe_var2.py --model_id 1 --algorithm teaspoon
# python pipe_var2.py --model_id 2 --algorithm teaspoon