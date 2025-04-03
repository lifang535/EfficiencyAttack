cd ..

export CUDA_VISIBLE_DEVICES=2

python pipe.py --model_id 0 --algorithm teaspoon --target_idx 23
python pipe.py --model_id 1 --algorithm teaspoon --target_idx 23
python pipe.py --model_id 2 --algorithm teaspoon --target_idx 23

python pipe.py --model_id 0 --algorithm teaspoon --target_idx 68
python pipe.py --model_id 1 --algorithm teaspoon --target_idx 68
python pipe.py --model_id 2 --algorithm teaspoon --target_idx 68

python pipe.py --model_id 0 --algorithm teaspoon
python pipe.py --model_id 1 --algorithm teaspoon
python pipe.py --model_id 2 --algorithm teaspoon