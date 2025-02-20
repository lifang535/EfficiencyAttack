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
```

```sh
# to measure the actual computational cost
ncu --metrics \
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum \
  python your_script.py
```