cd ..

# GPU 6 block
CUDA_VISIBLE_DEVICES=6 bash -c '
python main.py --model_id 0 --algorithm tea_400 --it_num 400 --val_size 100 --target_idx 0 --to_save_list 399 & 
python main.py --model_id 1 --algorithm tea_400 --it_num 400 --val_size 100 --target_idx 0 --to_save_list 399 & 
python main.py --model_id 2 --algorithm tea_400 --it_num 400 --val_size 100 --target_idx 0 --to_save_list 399 & 
wait
' &

# GPU 7 block
CUDA_VISIBLE_DEVICES=7 bash -c '
python main.py --model_id 0 --algorithm tea_100 --it_num 100 --val_size 100 --target_idx 0 --to_save_list 99 & 
python main.py --model_id 1 --algorithm tea_100 --it_num 100 --val_size 100 --target_idx 0 --to_save_list 99 & 
python main.py --model_id 2 --algorithm tea_100 --it_num 100 --val_size 100 --target_idx 0 --to_save_list 99 & 
wait
' &

wait
echo "job done"

# bash -c '
# python main.py --model_id 0 --algorithm eps_2 --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# python main.py --model_id 1 --algorithm eps_2 --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# python main.py --model_id 2 --algorithm eps_2 --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199

# python main.py --model_id 0 --algorithm eps_8 --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# python main.py --model_id 1 --algorithm eps_8 --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# python main.py --model_id 2 --algorithm eps_8 --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# ' &

# bash -c '
# python main.py --model_id 0 --algorithm norm --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# python main.py --model_id 1 --algorithm norm --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# python main.py --model_id 2 --algorithm norm --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199

# python main.py --model_id 0 --algorithm area --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# python main.py --model_id 1 --algorithm area --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# python main.py --model_id 2 --algorithm area --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# ' &

# bash -c '
# python main.py --model_id 0 --algorithm norm_and_area --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# python main.py --model_id 1 --algorithm norm_and_area --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199
# python main.py --model_id 2 --algorithm norm_and_area --it_num 200 --val_size 100 --target_idx 0 --to_save_list 199

# python main.py --model_id 0 --algorithm tea_100 --it_num 100 --val_size 100 --target_idx 0 --to_save_list 99
# python main.py --model_id 1 --algorithm tea_100 --it_num 100 --val_size 100 --target_idx 0 --to_save_list 99
# python main.py --model_id 2 --algorithm tea_100 --it_num 100 --val_size 100 --target_idx 0 --to_save_list 99
# ' 

