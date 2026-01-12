import torch
import random
import json
import os, glob
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

def seed_everything(seed=2026):
    """
    Set all random seeds to the same value for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clip_input(tokenizer, full_prompt, end_prompt, max_new_tokens=512, tree_length=250, max_output_length=4096):
    inputs = tokenizer(full_prompt, return_tensors='pt').to("cuda")
    end_prompt_length = len(tokenizer(end_prompt, return_tensors='pt').input_ids[0])
    input_ids = inputs.input_ids
    if len(input_ids[0]) + max_new_tokens + tree_length >= max_output_length:
        sample_num = (len(input_ids[0]) + max_new_tokens + tree_length - max_output_length)
        input_ids = torch.cat((input_ids[0][:-(end_prompt_length+sample_num)], input_ids[0][-end_prompt_length:]), dim=0).unsqueeze(0)
    return input_ids


def load_data(model_id, task_name, seed, data_num=10, sub_task_name=None):
    chat_instruction = ""
    if task_name == 'cnndm':
        chat_instruction = "Summarize the given text accurately and concisely."
        n_shot = 1
        data = load_dataset('cnn_dailymail', name='3.0.0', split='test').shuffle(seed=seed).select(range(data_num))
        shots = load_dataset('cnn_dailymail', name='3.0.0', split='train').shuffle(seed=seed).select(range(n_shot))
        prompt_keys = ['article', 'highlights']
        instructions = ['Article: ', '\nSummary: ']
    elif task_name == 'gsm8k':
        chat_instruction = "Solve math problems step-by-step with clear reasoning."
        n_shot = 5
        data = load_dataset('gsm8k', name='main', split='test').shuffle(seed=seed).select(range(data_num))
        shots = load_dataset('gsm8k', name='main', split='train').shuffle(seed=seed).select(range(n_shot))
        prompt_keys = ['question', 'answer']
        instructions = ['Question: ', '\nAnswer: ']
    elif task_name == 'tinystories':
        chat_instruction = "Write a coherent and engaging story based on the prompt."
        n_shot = 0
        data = load_dataset("roneneldan/TinyStories", name='default', split='validation').shuffle(seed=seed).select(range(data_num))
    elif task_name == 'nq':
        n_shot = 1
        data = load_dataset('rojagtap/natural_questions_clean', name='long', split='validation[10%:]').shuffle(
            seed=seed).select(range(data_num))
        shots = load_dataset('rojagtap/natural_questions_clean', name='long', split='validation[:10%]').shuffle(
            seed=seed).select(range(n_shot))
        prompt_keys = ['question', 'long_answer_candidates']
        instructions = ['Question: ', '\nAnswer: ']
    elif task_name == 'QA':
        chat_instruction = "Answer the following questions based on the given choices."
        n_shot = 1
        data = load_dataset('tau/commonsense_qa', split='test').shuffle(seed=seed).select(range(data_num))
        shots = load_dataset('tau/commonsense_qa', split='train').shuffle(seed=seed).select(range(n_shot))
        prompt_keys = ['question', 'choices', 'answerKey']
        instructions = ['Question: ', '\nChoices: ', '\nAnswer: ']
    elif task_name == 'xsum':
        chat_instruction = "Summarize the given text in only one sentence."
        n_shot = 5
        data = load_dataset('xsum', split='test').shuffle(seed=seed).select(range(data_num))
        shots = load_dataset('xsum', split='train').shuffle(seed=seed).select(range(n_shot))
        prompt_keys = ['document', 'summary']
        instructions = ['Article: ', '\nSummary: ']
    elif task_name == 'wmt16':
        chat_instruction = "Translate German text into fluent and accurate English."
        n_shot = 5
        data = load_dataset('wmt16', name='de-en', split='test').shuffle(seed=seed).select(range(data_num))
        shots = load_dataset('wmt16', name='de-en', split='train').shuffle(seed=seed).select(range(n_shot))
        prompt_keys = ['de', 'en']
        instructions = ['Translate German to English: ', '\nAnswer: ']
    elif task_name == 'math':
        chat_instruction = "Solve math problems step-by-step with clear reasoning."
        n_shot = 5
        task_path = f'/your/path/to/datasets/MATH/'
        random.seed(seed)

        json_count = len(glob.glob(os.path.join(task_path, f'test/{sub_task_name}/*.json')))
        random_datas = [random.randint(0, json_count-1) for _ in range(data_num)]
        data = []
        for random_data in random_datas:
            data_path = os.path.join(task_path, f'test/{sub_task_name}/{random_data}.json')
            with open(data_path, 'r') as f:
                json_data = json.load(f) 
                data.append(json_data)

        json_count = len(glob.glob(os.path.join(task_path, f'train/{sub_task_name}/*.json')))
        random_shots = [random.randint(0, json_count-1) for _ in range(n_shot)]
        shots = []
        for random_shot in random_shots:
            shot_path = os.path.join(task_path, f'train/{sub_task_name}/{random_shot}.json')
            with open(shot_path, 'r') as f:
                json_data = json.load(f) 
                shots.append(json_data)

        prompt_keys = ['problem', 'solution']
        instructions = ['Question: ', '\nAnswer: ']
    elif task_name == 'sql':
        chat_instruction = "Convert natural language into correct SQL queries."
        n_shot = 1
        task_path = f'/your/path/to/datasets/Spider2/'
        random.seed(seed)

        json_count = len(glob.glob(os.path.join(task_path, 'test/*.json')))
        random_datas = [random.randint(1, json_count) for _ in range(data_num)]
        data = []
        for random_data in random_datas:
            data_path = os.path.join(task_path, f'test/{random_data}.json')
            with open(data_path, 'r') as f:
                json_data = json.load(f) 
                data.append(json_data)

        json_count = len(glob.glob(os.path.join(task_path, 'train/*.json')))
        random_shots = [random.randint(1, json_count) for _ in range(n_shot)]
        shots = []
        for random_shot in random_shots:
            shot_path = os.path.join(task_path, f'train/{random_shot}.json')
            with open(shot_path, 'r') as f:
                json_data = json.load(f) 
                shots.append(json_data)

        prompt_keys = ['instruction', 'content']
        instructions = ['Instruction: ', 'SQL Query: ']
    elif task_name == 'alpaca':
        chat_instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        n_shot = 5
        data = load_dataset('tatsu-lab/alpaca', split='train[10%:]').shuffle(seed=seed).select(range(data_num))
        shots = load_dataset('tatsu-lab/alpaca', split='train[:10%]').shuffle(seed=seed).select(range(n_shot))
        prompt_keys = ['instruction', 'output']
        instructions = ['Instruction: ', '\nAnswer: ']
    else:
        n_shot = 5
        data = load_dataset('tatsu-lab/alpaca', split='train[10%:]').shuffle(seed=seed).select(range(data_num))
        shots = load_dataset('tatsu-lab/alpaca', split='train[:10%]').shuffle(seed=seed).select(range(n_shot))
        prompt_keys = ['instruction', 'output']
        instructions = ['Below is an instruction that describes a task. Write a response that appropriately completes'
                        ' the request.\n\n### Instruction:\n', '\n\n### Response:\n']
    prompt_shots = ''
    # instruct model use zero-shot
    if not model_id.endswith("chat") and not model_id.endswith("instruct"):
        for i in range(n_shot):
            if task_name == 'wmt16':
                prompt = (instructions[0] + shots[i]['translation'][prompt_keys[0]]
                        + instructions[1] + shots[i]['translation'][prompt_keys[1]].replace('\n', '') + '\n')
            elif task_name == 'QA':
                QAchoice = ''
                for choice, text in zip(shots[i][prompt_keys[1]]['label'], shots[i][prompt_keys[1]]['text']):
                    QAchoice += f"{choice}: {text}\n"
                prompt = (instructions[0] + shots[i][prompt_keys[0]] 
                        + instructions[1] + QAchoice
                        + instructions[2] + shots[i][prompt_keys[2]].replace('\n', '') + '\n')
            elif task_name == 'nq':
                prompt = instructions[0] + shots[i][prompt_keys[0]] + instructions[1] + shots[i][prompt_keys[1]][0].replace(
                    '\n', '') + '\n'
            else:
                prompt = instructions[0] + shots[i][prompt_keys[0]] + instructions[1] + shots[i][prompt_keys[1]].replace(
                    '\n', '') + '\n'
            prompt_shots += prompt

    prompts = []
    for sample in tqdm(data, desc=f"Generating {task_name} Prompts"):
        if task_name == 'cnndm':
            full_prompt = prompt_shots + 'Article: ' + sample['article'] + '\nSummary:'
            end_prompt = '\nSummary:'
        elif task_name == 'gsm8k' or task_name == 'nq':
            full_prompt = prompt_shots + 'Question: ' + sample['question'] + '\nAnswer:'
            end_prompt = '\nAnswer:'
        elif task_name == 'wmt16':
            full_prompt = prompt_shots + 'Translate German to English: ' + sample['translation']['de'] + '\nAnswer:'
            end_prompt = '\nAnswer:'
        elif task_name == 'humaneval':
            prompt = prompt['prompt'].replace("    ", "\t")
            full_prompt = prompt
            end_prompt = ''
        elif task_name == 'tinystories':
            full_prompt = 'Write a tiny story: ' + sample['text'] + '\nWrite a tiny story:'
            end_prompt = '\nWrite a tiny story:'
        elif task_name == 'xsum':
            full_prompt = prompt_shots + 'Article: ' + sample['document'] + '\nSummary:'
            end_prompt = '\nSummary:'
        elif task_name == 'math':
            full_prompt = prompt_shots + 'Question: ' + sample['problem'] + '\nAnswer:'
            end_prompt = '\nAnswer:'
        elif task_name == 'sql':
            full_prompt = prompt_shots + 'Instruction: ' + sample['instruction'] + '\nSQL Query:'
            end_prompt = '\nSQL Query:'
        elif task_name == 'alpaca':
            full_prompt = prompt_shots + 'Instruction: '+ sample['instruction'] + '\n' + sample['input'] + '\nAnswer:'
            end_prompt = '\n\n### Response:\n'
        elif task_name == 'QA':
            QAchoice = ''
            for choice, text in zip(shots[i][prompt_keys[1]]['label'], shots[i][prompt_keys[1]]['text']):
                QAchoice += f"{choice}: {text}\n"
            full_prompt = prompt_shots + 'Question: ' + sample['question'] + '\nChoices: ' + QAchoice + '\nAnswer:'
            end_prompt = '\nAnswer:'
        else:
            full_prompt = prompt_shots + 'Below is an instruction that describes a task. Write a response that appropriately ' +\
                            'completes the request.\n\n### Instruction:\n'+ prompt['instruction'] + '\n\n### Response:\n'
            end_prompt = '\n\n### Response:\n'
        prompts.append((full_prompt, end_prompt))

    return chat_instruction, prompts

def load_multiple_tasks(tokenizer, model_id, task_names, seed=2024, sub_task_name=None, mix_ratio=0, data_per_task=20):
    """
    1. Load multiple tasks and generate prompts for each task.
    2. Shuffle the tasks and mix them according to the mix_ratio.
    3. Return the mixed prompts.
    """
    all_prompts = []
    task_prompts = {}
    print('Generating prompts')

    for task_name in task_names:
        chat_instruction, prompts = load_data(model_id, task_name, seed, data_per_task, sub_task_name)
        
        # Support Llama-2-13b-chat and Qwen-2.5-14b-instruct
        if model_id.endswith("chat"):
            for i in range(len(prompts)):
                full_prompt, end_prompt = prompts[i]
                messages = [
                    {"role": "system", "content": chat_instruction},
                    {"role": "user", "content": full_prompt},
                ]
                full_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts[i] = (
                    full_prompt,
                    f"{end_prompt} [/INST]"
                )
        elif model_id.endswith("instruct"):
            for i in range(len(prompts)):
                full_prompt, end_prompt = prompts[i]
                messages = [
                    {"role": "user", "content": full_prompt},
                ]
                full_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts[i] = (
                    full_prompt,
                    f"{end_prompt} <|im_end|>\n<|im_start|>assistant"
                )
        
        task_prompts[task_name] = prompts
    
    last_task = None
    while any(task_prompts.values()):

        available_tasks = [
            task for task in task_prompts.keys()
            if task_prompts[task] and task != last_task
        ]
        if not available_tasks:
            available_tasks = [task for task in task_prompts.keys() if task_prompts[task]]
        
        task_name = random.choice(available_tasks)

        while task_prompts[task_name]:
            all_prompts.append(task_prompts[task_name].pop(0)) 
            # mix_ratio=0 menas no mixing, mix_ratio=1 means all mixing
            if random.random() < mix_ratio:    
                break
        
        last_task = task_name
    
    print('Prompts are generated')
    return all_prompts