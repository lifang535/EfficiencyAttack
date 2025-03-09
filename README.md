##### To perform efficiency attack on DETR models

```sh
conda create -n teaspoon python=3.9.21
conda activate teaspoon
```

```sh 
export HF_HOME="<your_huggingface_cache_directory>"
export TRANSFORMERS_CACHE="<your_huggingface_cache_directory>"

# for example:

export HF_HOME="~/huggingface_cache"
export TRANSFORMERS_CACHE="~/huggingface_cache"
```

```sh
git clone git@github.com:<replace_this_with_the_actual_name_of_repo>.git && cd <replace_this_with_the_actual_name_of_repo>
git checkout teaspoon
pip install -r requirements.txt
```

```sh
cd scripts
./baseline.sh
./teaspoon.sh
./teastatic.sh
```

or


```sh
# need to at least specify these two parameters
python ../main.py --model_id <id> --algorithm <a> 

# also other optional parameters
python ../main.py --model_id <id> --algorithm <a> --it_num <n> --val_size <val> --target_idx <t> --output_dir <o_dir> --if_save <s> --save_dir <s_dir>
```


##### To run the pipeline

```sh
conda create -n teapipe python=3.12.9
conda activate teapipe
```

```sh
cd pipeline_traffic
pip install pipeline_requirements.txt
```

```sh
alias python=python3 # if necessary
python traffic.py --model_id 0 --algorithm teaspoon
```
