LLAMA_PATH=/your/path/to/llama-2-13b # llama-2-13b, llama-2-13b-chat, qwen-2.5-14b, qwen-2.5-14b-instruct
MODEL_NAME=llama-2-13b # llama-2-13b, llama-2-13b-chat, qwen-2.5-14b, qwen-2.5-14b-instruct
TASK_NAME="cnndm" # cnndm, gsm8k, wmt16, tinystories, sql, math, alpaca, all
SUB_TASK_NAME="algebra" # algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus
DATA_NUM=1000
SEED=2026
GPU_DEVICES=0,1
MAX_NEW_TOKENS=32
METHOD="tsne"   # "umap" or "tsne"
CLUSTER_NUM=10
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

# Use to get last hidden vectors for various tasks
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m knn.last_hidden_state_utils --model-path $LLAMA_PATH --model-id ${MODEL_NAME} --max-new-tokens ${MAX_NEW_TOKENS} --task-name ${TASK_NAME} --sub-task-name ${SUB_TASK_NAME} --data-num ${DATA_NUM} --seed ${SEED} --dtype ${torch_dtype}

# Use to get knn results
python -m knn.visualization --model-id ${MODEL_NAME} --task-name ${TASK_NAME} --data-num ${DATA_NUM} --method ${METHOD} --cluster-num ${CLUSTER_NUM}

# Use to get knn anchors
python -m knn.knn_utils --model-id ${MODEL_NAME} --task-name ${TASK_NAME} --sub-task-name ${SUB_TASK_NAME} --cluster-num ${CLUSTER_NUM}

