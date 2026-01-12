LLAMA_PATH=/your/path/to/llama-2-13b # llama-2-13b, llama-2-13b-chat, qwen-2.5-14b, qwen-2.5-14b-instruct
MODEL_NAME=llama-2-13b # llama-2-13b, llama-2-13b-chat, qwen-2.5-14b, qwen-2.5-14b-instruct
TEMP=0.0 # 0.2 for general tasks and 0.6 for code generation
TOP_P=0.85 # 0.85 for general tasks and 0.95 for code generation
GPU_DEVICES=0,1
MAX_NEW_TOKENS=1024
SEED=2026
MIX_RATIO=1.0 # The probability of the next sampling coming from different tasks
TASK_NAME="all" # cnndm, gsm8k, wmt16, tinystories, sql, math, xsum, QA, alpaca, all(not include xsum, QA and alpaca)
DATA_NUM=30 # data number for each task

torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_knn --model-path $LLAMA_PATH --model-id ${MODEL_NAME} --max-new-tokens ${MAX_NEW_TOKENS} --task-name ${TASK_NAME} --data-num ${DATA_NUM} --mix-ratio ${MIX_RATIO} --seed ${SEED} --dtype ${torch_dtype}
