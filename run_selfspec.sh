LLAMA_PATH=/your/path/to/llama-2-13b # llama-2-13b, llama-2-13b-chat, qwen-2.5-14b, qwen-2.5-14b-instruct
MODEL_NAME=qwen-2.5-14b-instruct # llama-2-13b, llama-2-13b-chat, qwen-2.5-14b, qwen-2.5-14b-instruct
TEMP=0.0 # 0.2 for general tasks and 0.6 for code generation
TOP_P=0.85 # 0.85 for general tasks and 0.95 for code generation
GPU_DEVICES=0,1
MAX_NEW_TOKENS=64
SEED=2026
MIX_RATIO=0.0 # The probability of the next sampling coming from different tasks
TASK_NAME="cnndm" # cnndm, gsm8k, wmt16, tinystories, sql, math, mixed, xsum, QA
SUB_TASK_NAME="mix" # only for math task: algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalulus; default is none
DATA_NUM=30 # data number for each task

torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_selfspec --model-path $LLAMA_PATH --model-id ${MODEL_NAME} --max-new-tokens ${MAX_NEW_TOKENS} --task-name ${TASK_NAME} --sub-task-name ${SUB_TASK_NAME} --data-num ${DATA_NUM} --mix-ratio ${MIX_RATIO} --seed ${SEED} --dtype ${torch_dtype}
