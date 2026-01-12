Llama_PATH=/your/path/to/llama-2-13b
MODEL_NAME=llama-2-13b # llama-2-13b, llama-2-13b-chat, qwen-2.5-14b, qwen-2.5-14b-instruct
GPU_DEVICES=0,1
TASK_NAME="cnndm" # cnndm, gsm8k, wmt16, tinystories, sql, math, mixed
SEARCH_ITER=1000
MAX_NEW_TOKENS=32
DATA_NUM=8
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python searching.py --model-path ${Llama_PATH} --model-id ${MODEL_NAME} --task ${TASK_NAME} --search-iter ${SEARCH_ITER} --max-new-tokens ${MAX_NEW_TOKENS} --data-num ${DATA_NUM}