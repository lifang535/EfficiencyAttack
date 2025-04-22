cd ..

CUDA_VISIBLE_DEVICES=0 bash -c '
python pipe_var2.py --model_id 0 --algorithm weighted --target_idx 0 2 &
python pipe_var2.py --model_id 0 --algorithm unweighted --target_idx 0 2 &
wait
' &

CUDA_VISIBLE_DEVICES=1 bash -c '
python pipe_var1.py --model_id 0 --algorithm weighted --target_idx 0 2 &
python pipe_var1.py --model_id 0 --algorithm unweighted --target_idx 0 2 &
wait
' &

CUDA_VISIBLE_DEVICES=2 bash -c '
python pipe.py --model_id 0 --algorithm weighted --target_idx 0 2 &
python pipe.py --model_id 0 --algorithm unweighted --target_idx 0 2 &
wait
' &

wait
echo "job done"