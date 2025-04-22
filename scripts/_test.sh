cd ..

bash -c '
python main.py --model_id 0 --algorithm teaspoon --it_num 200 --val_size 100 --target_idx 20 --to_save_list 199 &
python main.py --model_id 0 --algorithm teaspoon --it_num 200 --val_size 100 --target_idx 72 --to_save_list 199 &
python main.py --model_id 0 --algorithm teaspoon --it_num 200 --val_size 100 --target_idx 20 23 --to_save_list 199 
wait
' 

wait
echo "job done"