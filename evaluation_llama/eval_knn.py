"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import logging
import os
import time
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm
from evaluation_llama.utils import clip_input, load_multiple_tasks, seed_everything


def run_eval_knn(
        model,
        tokenizer,
        forward_func,
        model_id,
        answer_file,
        max_new_tokens,
        num_gpus_per_model,
        num_gpus_total,
        task_names,
        data_num,
        mix_ratio,
        seed,
        **kwargs,
):
    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0

    seed_everything(seed)

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(os.path.expanduser(answer_file), "w") as fout:
        ans_json = {
            "model_id": model_id,
            "tasks": task_names,
            "data_per_task": data_num,
            "mix_ratio": mix_ratio,
            "seed": seed
        }
        fout.write(json.dumps(ans_json) + "\n")

    prompts = load_multiple_tasks(tokenizer, model_id, task_names, seed, mix_ratio=mix_ratio, data_per_task=data_num, sub_task_name='mix')
    get_answers_func = get_model_answers_knn

    get_answers_func(
        model,
        tokenizer,
        forward_func,
        model_id,
        prompts,
        answer_file,
        max_new_tokens,
        **kwargs,
    )


@torch.inference_mode()
def get_model_answers_knn(
        model,
        tokenizer,
        forward_func,
        model_id,
        prompts,
        answer_file,
        max_new_tokens,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    accept_lengths_tree = []
    total_draft_num = 0

    knn_vectors = torch.tensor(np.load(f'data/{model_id}/all_tasks_last_hidden_vector.npy')).to("cuda")
    knn_labels = np.load(f'data/{model_id}/all_tasks_label.npy')
    knn_vectors_cpu = knn_vectors.cpu().numpy()
    knn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
    knn_model.fit(knn_vectors_cpu)
    print('KNN model loaded')

    with open(f'{model_id}_skip.json', "r") as file:
        skip_layer_set = json.load(file)

    for full_prompt, end_prompt in tqdm(prompts):
        choices = []
        input_ids = clip_input(tokenizer, full_prompt, end_prompt, max_new_tokens=max_new_tokens,
                               max_output_length=model.config.max_position_embeddings)
        cur_accept_lengths_tree = []
        cur_draft_num = 0
        steps = []
        new_tokens = []
        wall_time = []
        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token_num, step, accept_length_tree, draft_token_num, assigned_task, knn_time = forward_func(
            input_ids,
            model,
            model_id,
            knn_model,
            knn_labels,
            skip_layer_set,
            tokenizer,
            max_new_tokens,
            **kwargs,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        cur_accept_lengths_tree.extend(accept_length_tree)
        cur_draft_num += draft_token_num
        output_ids = output_ids[0][len(input_ids[0]):]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        steps.append(int(step))
        new_tokens.append(int(new_token_num))
        wall_time.append(total_time)

        accept_lengths_tree.extend(cur_accept_lengths_tree)
        total_draft_num += cur_draft_num
        choices.append({"knn_result:": assigned_task ,"turns": output, "decoding_steps": steps, "new_tokens": new_tokens, 
                        "wall_time": wall_time, "knn_time":knn_time, "accept_lengths": cur_accept_lengths_tree,
                        "acceptance_rate": (sum(cur_accept_lengths_tree) - len(
                            cur_accept_lengths_tree)) / cur_draft_num})

        # Dump answers
        # os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
        # break
    mean_accepted_tokens = np.mean(accept_lengths_tree)
    if mean_accepted_tokens > 1:
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "Mean accepted tokens": np.mean(accept_lengths_tree),
                "Token acceptance rate": (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num,
            }
            fout.write(json.dumps(ans_json) + "\n")
            print("#Mean accepted tokens:", np.mean(accept_lengths_tree))
            print("Token acceptance rate:", (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num)
