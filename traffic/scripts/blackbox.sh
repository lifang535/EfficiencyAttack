cd ..

CUDA_VISIBLE_DEVICES=0 bash -c '
python pipe.py --model_id 1 --algorithm teaspoon --target_idx 2 --ps_path blackbox  &
python pipe.py --model_id 1 --algorithm teaspoon --target_idx 0 --ps_path blackbox &
wait
' &


CUDA_VISIBLE_DEVICES=1 bash -c '
python pipe.py --model_id 0 --algorithm teaspoon &
python pipe_var1.py --model_id 0 --algorithm teaspoon &
python pipe_var2.py --model_id 0 --algorithm teaspoon &
wait
' &

CUDA_VISIBLE_DEVICES=2 bash -c '
python pipe.py --model_id 1 --algorithm teaspoon &
python pipe_var1.py --model_id 1 --algorithm teaspoon &
python pipe_var2.py --model_id 1 --algorithm teaspoon &
wait
' &

CUDA_VISIBLE_DEVICES=3 bash -c '
python pipe.py --model_id 2 --algorithm teaspoon &
python pipe_var1.py --model_id 2 --algorithm teaspoon &
python pipe_var2.py --model_id 2 --algorithm teaspoon &
wait
' &

wait
echo "job done"