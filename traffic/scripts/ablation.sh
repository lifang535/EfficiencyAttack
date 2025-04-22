cd ..

CUDA_VISIBLE_DEVICES=6 bash -c '
python pipe.py --model_id 0 --algorithm tea_400 --target_idx 0 &
python pipe.py --model_id 0 --algorithm tea_100 --target_idx 0 &
wait
python pipe.py --model_id 0 --algorithm area --target_idx 0 &
python pipe.py --model_id 0 --algorithm norm --target_idx 0 &
' &

CUDA_VISIBLE_DEVICES=7 bash -c '
python pipe.py --model_id 0 --algorithm eps_2 --target_idx 0 &
python pipe.py --model_id 0 --algorithm eps_8 --target_idx 0 &
wait
python pipe.py --model_id 0 --algorithm norm_and_area --target_idx 0 &
' &
