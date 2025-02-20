#!/bin/bash

# run targeted attack on following targets:
#   - person: 0
#   - car: 2
#   - microwave oven: 68
#   - giraffe: 23

python ../main.py --model_id 0 --algorithm teaspoon --it_num 200 --val_size 1000 --target_idx 0
python ../main.py --model_id 1 --algorithm teaspoon --it_num 200 --val_size 1000 --target_idx 0
python ../main.py --model_id 2 --algorithm teaspoon --it_num 200 --val_size 1000 --target_idx 0

python ../main.py --model_id 0 --algorithm teaspoon --it_num 200 --val_size 1000 --target_idx 2
python ../main.py --model_id 1 --algorithm teaspoon --it_num 200 --val_size 1000 --target_idx 2
python ../main.py --model_id 2 --algorithm teaspoon --it_num 200 --val_size 1000 --target_idx 2

python ../main.py --model_id 0 --algorithm teaspoon --it_num 200 --val_size 1000 --target_idx 23
python ../main.py --model_id 1 --algorithm teaspoon --it_num 200 --val_size 1000 --target_idx 23
python ../main.py --model_id 2 --algorithm teaspoon --it_num 200 --val_size 1000 --target_idx 23