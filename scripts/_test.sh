#!/bin/bash

#run baseline attack algorithm:
# - overlaod
# - phantom
# - slowtrack

python ../main.py --model_id 0 --algorithm overload --it_num 10 --val_size 10
python ../main.py --model_id 1 --algorithm overload --it_num 10 --val_size 10
python ../main.py --model_id 2 --algorithm overload --it_num 10 --val_size 10


python ../main.py --model_id 0 --algorithm phantom --it_num 10 --val_size 10
python ../main.py --model_id 1 --algorithm phantom --it_num 10 --val_size 10
python ../main.py --model_id 2 --algorithm phantom --it_num 10 --val_size 10


python ../main.py --model_id 0 --algorithm slowtrack --it_num 10 --val_size 10
python ../main.py --model_id 1 --algorithm slowtrack --it_num 10 --val_size 10
python ../main.py --model_id 2 --algorithm slowtrack --it_num 10 --val_size 10