#!/bin/bash
cd ..

python main.py --model_id 0 --algorithm overload 
python main.py --model_id 0 --algorithm phantom 
python main.py --model_id 0 --algorithm slowtrack 

python main.py --model_id 1 --algorithm overload 
python main.py --model_id 1 --algorithm phantom 
python main.py --model_id 1 --algorithm slowtrack 

python main.py --model_id 2 --algorithm overload 
python main.py --model_id 2 --algorithm phantom 
python main.py --model_id 2 --algorithm slowtrack 