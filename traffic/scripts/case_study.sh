cd ..

CUDA_VISIBLE_DEVICES=0 bash -c '
python pipe_var2.py --model_id 0 --algorithm adv --target_idx 0 2 &
wait
' &


CUDA_VISIBLE_DEVICES=1 bash -c '
python pipe_var2.py --model_id 0 --algorithm adv_cap --target_idx 0 2 &
wait
' &

CUDA_VISIBLE_DEVICES=2 bash -c '
python pipe_var2.py --model_id 0 --algorithm clean --target_idx 0 2 &
wait
' &

CUDA_VISIBLE_DEVICES=3 bash -c '
python pipe_var2.py --model_id 0 --algorithm adv_unweighted --target_idx 0 2 &
wait
' &

CUDA_VISIBLE_DEVICES=4 bash -c '
python pipe_var2.py --model_id 0 --algorithm adv_od --target_idx 0 2 &
wait
' &


wait
echo "job done"