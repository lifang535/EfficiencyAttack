
cd ..

export CUDA_VISIBLE_DEVICES=0

python pipe.py --model_id 0 --algorithm overload
python pipe.py --model_id 1 --algorithm overload
python pipe.py --model_id 2 --algorithm overload

python pipe.py --model_id 0 --algorithm phantom
python pipe.py --model_id 1 --algorithm phantom
python pipe.py --model_id 2 --algorithm phantom

python pipe.py --model_id 0 --algorithm slowtrack
python pipe.py --model_id 1 --algorithm slowtrack
python pipe.py --model_id 2 --algorithm slowtrack
