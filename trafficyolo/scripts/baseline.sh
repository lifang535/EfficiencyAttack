
#!/bin/bash

cd ..

CUDA_VISIBLE_DEVICES=0 bash -c '
python pipe.py --model_id 0 --algorithm overload &
python pipe.py --model_id 0 --algorithm phantom &
wait

python pipe.py --model_id 0 --algorithm slowtrack &
wait
' &

CUDA_VISIBLE_DEVICES=1 bash -c '
python pipe_var1.py --model_id 0 --algorithm overload &
python pipe_var1.py --model_id 0 --algorithm phantom &
wait

python pipe_var1.py --model_id 0 --algorithm slowtrack &
wait
' &

CUDA_VISIBLE_DEVICES=2 bash -c '
python pipe_var2.py --model_id 0 --algorithm overload &
python pipe_var2.py --model_id 0 --algorithm phantom &
wait

python pipe_var2.py --model_id 0 --algorithm slowtrack &
wait
' &

CUDA_VISIBLE_DEVICES=3 bash -c '
python pipe.py      --model_id 0 --algorithm clean &
python pipe_var1.py --model_id 0 --algorithm clean &
wait

python pipe_var2.py --model_id 0 --algorithm clean &
wait
' &

wait
echo "job done"
