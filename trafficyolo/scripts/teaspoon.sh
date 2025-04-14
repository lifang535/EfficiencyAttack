
#!/bin/bash

cd ..

CUDA_VISIBLE_DEVICES=4 bash -c '
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 0 &
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 2 &
wait

python pipe.py --model_id 0 --algorithm teaspoon --target_idx 0 2&
python pipe.py --model_id 0 --algorithm teaspoon &
wait
' &

CUDA_VISIBLE_DEVICES=5 bash -c '
python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 0 &
python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 2 &
wait

python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 0 2 &
python pipe_var1.py --model_id 0 --algorithm teaspoon &
wait
' &

CUDA_VISIBLE_DEVICES=6 bash -c '
python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0 &
python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 2 &
wait

python pipe_var2.py --model_id 0 --algorithm teaspoon --target_idx 0 2 &
python pipe_var2.py --model_id 0 --algorithm teaspoon &
wait
' &

wait
echo "job done"
