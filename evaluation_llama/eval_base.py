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

from tqdm import tqdm
from evaluation_llama.utils import clip_input, load_multiple_tasks, seed_everything


def run_eval(
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
        sub_task_name,
        baseline=False,
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
    
    with open(f'{model_id}_skip.json', "r") as f:
        skip_config = json.load(f)

    prompts = load_multiple_tasks(tokenizer, model_id, task_names, seed, sub_task_name, mix_ratio=mix_ratio, data_per_task=data_num)
    get_answers_func = get_model_answers

    #with open('results/prompts.json', "w", encoding="utf-8") as f:
        #json.dump(prompts, f, ensure_ascii=False, indent=4)

    if baseline == False:
        if len(task_names) == 1:
            task_key = f"{task_names[0]}_{model_id}"
        else:
            task_key = f"mixed_{model_id}"
        if task_key in skip_config:
            attn_skip_layer_id_set = skip_config[task_key].get("attention", [])
            mlp_skip_layer_id_set = skip_config[task_key].get("mlp", [])
            model.set_skip_layers(attn_skip_layer_id_set, mlp_skip_layer_id_set)
        else:
            attn_skip_layer_id_set = [1, 3, 5, 7 ,9, 11, 13, 15 ,17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37]
            mlp_skip_layer_id_set = [1, 3, 5, 7 ,9, 11, 13, 15 ,17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37]
            model.set_skip_layers(attn_skip_layer_id_set, mlp_skip_layer_id_set)

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
def get_model_answers(
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
        output_ids, new_token_num, step, accept_length_tree, draft_token_num = forward_func(
            input_ids,
            model,
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
        choices.append({"turns": output, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                        "accept_lengths": cur_accept_lengths_tree,
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
        best_attn_skip_layer_id_set, best_mlp_skip_layer_id_set = model.get_skip_layers()
        best_skip_ratio = (len(best_mlp_skip_layer_id_set) + len(best_attn_skip_layer_id_set)) / ((model.config.num_hidden_layers - 2) * 2)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "Mean accepted tokens": np.mean(accept_lengths_tree),
                "Token acceptance rate": (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num,
                "Skip Ratio": best_skip_ratio,
                "Attn Layer Set": best_attn_skip_layer_id_set,
                "MLP Layer Set": best_mlp_skip_layer_id_set,
            }
            fout.write(json.dumps(ans_json) + "\n")
            print("#Mean accepted tokens:", np.mean(accept_lengths_tree))
            print("Token acceptance rate:", (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num)
