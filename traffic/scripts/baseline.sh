
#!/bin/bash

cd ..

CUDA_VISIBLE_DEVICES=0 bash -c '
python pipe.py --model_id 0 --algorithm overload
python pipe.py --model_id 1 --algorithm overload
python pipe.py --model_id 2 --algorithm overload
' &

CUDA_VISIBLE_DEVICES=1 bash -c '
python pipe.py --model_id 0 --algorithm phantom
python pipe.py --model_id 1 --algorithm phantom
python pipe.py --model_id 2 --algorithm phantom
' &

CUDA_VISIBLE_DEVICES=2 bash -c '
python pipe.py --model_id 0 --algorithm slowtrack
python pipe.py --model_id 1 --algorithm slowtrack
python pipe.py --model_id 2 --algorithm slowtrack
' &

CUDA_VISIBLE_DEVICES=3 bash -c '
python pipe_var1.py --model_id 0 --algorithm overload
python pipe_var1.py --model_id 1 --algorithm overload
python pipe_var1.py --model_id 2 --algorithm overload
' &

CUDA_VISIBLE_DEVICES=4 bash -c '
python pipe_var1.py --model_id 0 --algorithm phantom
python pipe_var1.py --model_id 1 --algorithm phantom
python pipe_var1.py --model_id 2 --algorithm phantom
' &

CUDA_VISIBLE_DEVICES=5 bash -c '
python pipe_var1.py --model_id 0 --algorithm slowtrack
python pipe_var1.py --model_id 1 --algorithm slowtrack
python pipe_var1.py --model_id 2 --algorithm slowtrack
' &

CUDA_VISIBLE_DEVICES=6 bash -c '
python pipe_var2.py --model_id 0 --algorithm overload
python pipe_var2.py --model_id 1 --algorithm overload
python pipe_var2.py --model_id 2 --algorithm overload
' &

CUDA_VISIBLE_DEVICES=7 bash -c '
python pipe_var2.py --model_id 0 --algorithm phantom
python pipe_var2.py --model_id 1 --algorithm phantom
python pipe_var2.py --model_id 2 --algorithm phantom

python pipe_var2.py --model_id 0 --algorithm slowtrack
python pipe_var2.py --model_id 1 --algorithm slowtrack
python pipe_var2.py --model_id 2 --algorithm slowtrack
'
wait
echo "job done"
