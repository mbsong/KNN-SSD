import argparse
import numpy as np
import torch
import os
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.utils import str_to_torch_dtype
from tqdm import tqdm
from evaluation_llama.utils import clip_input, load_multiple_tasks, seed_everything


def get_last_hidden_vector(input_ids, model):
    """
    Get the last hidden state vector for the given input IDs.
    Args:
        input_ids: The input IDs for the model.
        model: The model to be used.
    Returns:
        last_hidden_vector: The last hidden state vector.
    """
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
    past_key_values = None

    with torch.no_grad():  
        output = model(input_ids, 
                        attention_mask=attention_mask, 
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True)
        last_hidden_state = output.hidden_states[-1] 
        last_hidden_vector = last_hidden_state.mean(dim=1)
        last_hidden_vector_cpu = last_hidden_vector.cpu().numpy()
    return last_hidden_vector_cpu


def run_model(
        model,
        model_id,
        tokenizer,
        save_path,
        max_new_tokens,
        num_gpus_per_model,
        num_gpus_total,
        task_name,
        sub_task_name,
        data_num,
        seed
):
    """
    Run the model to get the last hidden state vectors for the given task.
    Args:
        save_path: The path to save the output vectors.
        task_name: The name of the task (e.g., 'cnndm', 'gsm8k', etc.).
        sub_task_name: The sub-task name for MATH dataset.
        data_num: The number of samples to be generated.
    """
    assert num_gpus_total % num_gpus_per_model == 0
    seed_everything(seed)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    prompts = load_multiple_tasks(tokenizer, model_id, [task_name], seed, sub_task_name, data_per_task=data_num)
        
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    all_hidden_vectors = []

    for full_prompt, end_prompt in tqdm(prompts):
        input_ids = clip_input(tokenizer, full_prompt, end_prompt, max_new_tokens=max_new_tokens,
                               max_output_length=model.config.max_position_embeddings)
        torch.cuda.synchronize()
        last_hidden_vector = get_last_hidden_vector(input_ids, model)
        all_hidden_vectors.append(last_hidden_vector)
    
    all_hidden_vectors = np.vstack(all_hidden_vectors)
    np.save(save_path, all_hidden_vectors)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="The task name for the dataset. Now support 'cnndm', 'gsm8k', 'wmt16', 'tinystories', 'sql', and 'math'.",
    )
    parser.add_argument(
        "--sub-task-name",
        type=str,
        default=None,
        help="The sub-task name for MATH dataset.",
    )
    parser.add_argument(
        "--data-num",
        type=int,
        default=10,
        help="The number of samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="The sampling seed.",
    )

    args = parser.parse_args()

    if args.task_name == 'math':
        if args.sub_task_name is None:
            raise ValueError("Please specify a sub-task name for the MATH dataset.")
        save_path = f"data/{args.model_id}/math_{args.sub_task_name}_{args.data_num}_samples.npy"
    else:
        save_path = f"data/{args.model_id}/{args.task_name}_{args.data_num}_samples.npy"

    print(f"Output to {save_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    run_model(
        model=model,
        model_id=args.model_id,
        tokenizer=tokenizer,
        save_path=save_path,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        task_name=args.task_name,
        sub_task_name=args.sub_task_name,
        data_num=args.data_num,
        seed=args.seed
    )

    print(f'{args.data_num} of {args.task_name} samples for {args.model_id} are generated')