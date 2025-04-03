cd ..

export CUDA_VISIBLE_DEVICES=1

python pipe.py --model_id 0 --algorithm teaspoon --target_idx 0
python pipe.py --model_id 1 --algorithm teaspoon --target_idx 0
python pipe.py --model_id 2 --algorithm teaspoon --target_idx 0

python pipe.py --model_id 0 --algorithm teaspoon --target_idx 2
python pipe.py --model_id 1 --algorithm teaspoon --target_idx 2
python pipe.py --model_id 2 --algorithm teaspoon --target_idx 2

python pipe.py --model_id 0 --algorithm teaspoon --target_idx 0 2
python pipe.py --model_id 1 --algorithm teaspoon --target_idx 0 2
python pipe.py --model_id 2 --algorithm teaspoon --target_idx 0 2