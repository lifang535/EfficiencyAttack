#!/bin/bash
cd ..

CUDA_VISIBLE_DEVICES=0 bash -c '
python pipe.py --model_id 0 --algorithm teaspoon --target_idx 2 &
python pipe.py --model_id 2 --algorithm teaspoon --target_idx 2 &
' &

CUDA_VISIBLE_DEVICES=1 bash -c '
python pipe_var1.py --model_id 0 --algorithm teaspoon --target_idx 2 &
python pipe_var1.py --model_id 2 --algorithm teaspoon --target_idx 2 &
wait
'  &

wait

echo "job done"