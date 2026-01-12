import torch
import argparse
from transformers import AutoTokenizer
from decoding import infer
import os
import random
import json
from datasets import load_dataset
from bayes_opt import BayesianOptimization

class LayerSkippingSearching:
    def __init__(
        self,
        model,
        tokenizer,
        evaluate_prompts,
        evaluate_config={"generate_fn": "essg", "max_new_tokens": 32},
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.evaluate_prompts = evaluate_prompts
        self.evaluate_config = evaluate_config

        self.pbounds = {
            f"x{i}": (0, 1) for i in range(self.config.num_hidden_layers * 2)
        }

        self.optimizer = BayesianOptimization(
            f=self._black_box_evaluate_function, pbounds=self.pbounds, random_state=1, verbose=1
        )

        self.optimizer.set_gp_params(alpha=1e-2)

    def _black_box_evaluate_function(self, **kargs):
        attn_skip_layers = []
        for i in range(self.config.num_hidden_layers):
            if kargs[f"x{i}"] > 0.5:
                attn_skip_layers.append(i)
        mlp_skip_layers = []
        for i in range(
            self.config.num_hidden_layers, self.config.num_hidden_layers * 2
        ):
            if kargs[f"x{i}"] > 0.5:
                mlp_skip_layers.append(i - self.config.num_hidden_layers)

        self.model.set_skip_layers(
            attn_skip_layer_id_set=attn_skip_layers,
            mlp_skip_layer_id_set=mlp_skip_layers,
        )

        total_time = 0
        total_tokens = 0

        for prompt in self.evaluate_prompts:
            ret = infer(self.model, self.tokenizer, prompt, **self.evaluate_config)
            total_time += ret["time"]
            total_tokens += self.evaluate_config.get("max_new_tokens", 10)

        print(
            "Log:",
            total_tokens / total_time,
            "tokens/s",
            "Skipped attn:",
            len(attn_skip_layers),
            "Skipped mlp:",
            len(mlp_skip_layers),
        )

        return total_tokens / total_time

    def probe(self, attn_skip_layers, mlp_skip_layers):
        """
        Add some good points to accelerate searching
        """

        params = {f"x{i}": 0.0 for i in range(self.config.num_hidden_layers * 2)}
        for i in attn_skip_layers:
            params[f"x{i}"] = 1.0
        for i in mlp_skip_layers:
            params[f"x{i+self.config.num_hidden_layers}"] = 1.0
        self.optimizer.probe(params=params, lazy=True)

    def search(self, n_iter=1000):
        self.optimizer.maximize(init_points=0, n_iter=n_iter)
        return self.get_solution()

    def get_solution(self):

        skip_attn_layers = []
        for i in range(self.config.num_hidden_layers):
            if self.optimizer.max["params"][f"x{i}"] > 0.5:
                skip_attn_layers.append(i)

        skip_mlp_layers = []
        for i in range(
            self.config.num_hidden_layers, self.config.num_hidden_layers * 2
        ):
            if self.optimizer.max["params"][f"x{i}"] > 0.5:
                skip_mlp_layers.append(i - self.config.num_hidden_layers)

        return skip_attn_layers, skip_mlp_layers


def run_search(model_id, model_path, task='cnndm', search_iter=1000, max_new_tokens=32, data_num=8, file_path = None):
    torch.nn.Linear.reset_parameters = lambda x: None
    # Llama-2-13b needs two RTX3090-24G
    if model_id.startswith('qwen'):
        from modeling_qwen2 import Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(model_path, 
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto")
    else:
        from modeling_llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_path, 
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto")
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    chat_instruction = []
    prompts = []
    if task == 'cnndm':
        cnn = load_dataset('cnn_dailymail', '3.0.0').shuffle(4242)
        for i in range(data_num):
            item = cnn['train'][i + 100]
            cnn_context = 'Article: ' + item['article'] + '\nSummary: ' + item['highlights'].replace('\n', '')

            item = cnn['train'][i]
            prompt = cnn_context + '\nArticle: ' + item['article'] + '\nSummary:'
            prompts.append(prompt)
            chat_instruction.append("Summarize the given text accurately and concisely.")
    elif task == 'cnndm-1':
        json_file_path = 'cnndm_prompts.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data[:data_num]: 
            prompts.append(item[0])
            chat_instruction.append("Summarize the given text accurately and concisely.")
    elif task == 'cnndm-2':
        json_file_path = 'cnndm_prompts.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data[data_num:]: 
            prompts.append(item[0])
            chat_instruction.append("Summarize the given text accurately and concisely.")
    elif task == 'alpaca-1':
        json_file_path = 'alpaca_prompts.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = data[0:10]
        selected_data = random.sample(data, data_num)
        for item in selected_data: 
            prompts.append(item[0])
    elif task == 'alpaca-2':
        json_file_path = 'alpaca_prompts.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = data[10:20]
        selected_data = random.sample(data, data_num)
        for item in selected_data: 
            prompts.append(item[0])
    elif task == 'alpaca-3':
        json_file_path = 'alpaca_prompts.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = data[20:30]
        selected_data = random.sample(data, data_num)
        for item in selected_data: 
            prompts.append(item[0])
    elif task == 'alpaca-4':
        json_file_path = 'alpaca_prompts.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = data[30:40]
        selected_data = random.sample(data, data_num)
        for item in selected_data: 
            prompts.append(item[0])
    elif task == 'alpaca-5':
        json_file_path = 'alpaca_prompts.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = data[40:50]
        selected_data = random.sample(data, data_num)
        for item in selected_data: 
            prompts.append(item[0])
    elif task == 'gsm8k':
        gsm = load_dataset('gsm8k', name='main').shuffle(4242)
        for i in range(data_num):
            item = gsm['train'][i + 100]
            gsm8k_context = 'Question: ' + item['question'] + '\nAnswer: ' + item['answer'].replace('\n', '')

            item = gsm['train'][i]
            prompt = gsm8k_context + '\nQuestion: ' + item['question'] + '\nAnswer:'
            prompts.append(prompt)
            chat_instruction.append("Solve math problems step-by-step with clear reasoning.")
    elif task == 'alpaca':
        alpaca = load_dataset('tatsu-lab/alpaca').shuffle(4242)
        for i in range(data_num):
            item = alpaca['train'][i+100]
            alpaca_context = 'Instruction: ' + item['instruction'] + '\nResponse: ' + item['output'].replace('\n', '')
            
            item = alpaca['train'][i]
            prompt = alpaca_context + '\nInstruction: ' + item['instruction'] + '\nResponse:'
            prompts.append(prompt)
    elif task == 'wmt16':
        wmt16 = load_dataset('wmt16', name='de-en').shuffle(4242)
        for i in range(data_num):
            item = wmt16['train'][i+100]
            wmt16_context = 'Translate German to English: ' + item['translation']['de'] + '\nAnswer: ' + item['translation']['en'].replace('\n', '')
            
            item = wmt16['train'][i]
            prompt = wmt16_context + '\nTranslate German to English: ' + item['translation']['de'] + '\nAnswer: '
            prompts.append(prompt)
            chat_instruction.append("Translate German text into fluent and accurate English.")
    elif task == 'tinystories':
        tinystories = load_dataset("roneneldan/TinyStories", name='default', split='validation').shuffle(4242)
        for i in range(data_num):
            item = tinystories[i]
            prompt =  'Write a tiny story: ' + item['text'] + '\nWrite a tiny story:'
            prompts.append(prompt)
            chat_instruction.append("Write a coherent and engaging story based on the prompt.")
    elif task == 'nq':
        nq = load_dataset('rojagtap/natural_questions_clean', name='long', split='train[:200]').shuffle(4242)
        for i in range(data_num):
            item = nq[i+100]
            nq_context = 'Document: ' + item['document'] + 'Question: ' + item['question'] + '\nAnswer: ' + item['long_answer_candidates'][0].replace('\n', '')

            item = nq[i]
            prompt = nq_context + '\nQuestion: ' + item['question'] + '\nAnswer:'
            prompts.append(prompt)
    elif task == 'sql':
        task_path = '/home/mingbo/datasets/Spider2/'
        random.seed(2024)
        random_datas = [random.randint(1, 121) for _ in range(data_num)]
        random_shots = [random.randint(1, 426) for _ in range(data_num)]
        for i in range(data_num):
            file_name = f"train/{random_datas[i]}.json"
            sql_file_path = os.path.join(task_path, file_name)
            with open(sql_file_path, 'r') as f:
                item = json.load(f)
            sql_context = 'Instruction: ' + item['instruction'] + '\SQL Query: ' + item['content'].replace('\n', '')
            file_name = f"test/{random_shots[i]}.json"
            sql_file_path = os.path.join(task_path, file_name)
            with open(sql_file_path, 'r') as f:
                item = json.load(f)
            prompt = sql_context + '\nInstruction: ' + item['instruction'] + '\SQL Query:'
            prompts.append(prompt)
            chat_instruction.append("Convert natural language into correct SQL queries.")
    elif task == 'math':
        sub_task = 'number_theory'
        sub_task_path = f'/home/mingbo/datasets/MATH/train/{sub_task}'
        all_train_files = [file_name for file_name in os.listdir(sub_task_path) if file_name.endswith(".json")]
        random.seed(2024)
        selected_files = random.sample(all_train_files, min(len(all_train_files), 200))
        
        for i in range(data_num):
            file_name = selected_files[i+100]
            math_file_path = os.path.join(sub_task_path, file_name)
            with open(math_file_path, 'r') as f:
                item = json.load(f)
            math_context = 'Question: ' + item['problem'] + '\nAnswer: ' + item['solution'].replace('\n', '')

            file_name = selected_files[i]
            math_file_path = os.path.join(sub_task_path, file_name)
            with open(math_file_path, 'r') as f:
                item = json.load(f)
            prompt = math_context + '\nQuestion: ' + item['problem'] + '\nAnswer:'
            prompts.append(prompt)
            chat_instruction.append("Solve math problems step-by-step with clear reasoning.")
    elif task == 'mixed':

        cnn = load_dataset('cnn_dailymail', name='3.0.0', split='train').shuffle(4242).select(range(5))
        for i in range(2):
            item = cnn[i + 2]
            cnn_context = 'Article: ' + item['article'] + '\nSummary: ' + item['highlights'].replace('\n', '')

            item = cnn[i]
            prompt = cnn_context + '\nArticle: ' + item['article'] + '\nSummary:'
            prompts.append(prompt)
            chat_instruction.append("Summarize the given text accurately and concisely.") 
    
        gsm = load_dataset('gsm8k', name='main', split='train').shuffle(4242).select(range(5))
        for i in range(2):
            item = gsm[i + 2]
            gsm8k_context = 'Question: ' + item['question'] + '\nAnswer: ' + item['answer'].replace('\n', '')

            item = gsm[i]
            prompt = gsm8k_context + '\nQuestion: ' + item['question'] + '\nAnswer:'
            prompts.append(prompt)
            chat_instruction.append("Solve math problems step-by-step with clear reasoning.")

        wmt16 = load_dataset('wmt16', name='de-en', split='train').shuffle(4242).select(range(5))
        for i in range(2):
            item = wmt16[i + 2]
            wmt16_context = 'Translate German to English: ' + item['translation']['de'] + '\nAnswer: ' + item['translation']['en'].replace('\n', '')
            
            item = wmt16[i]
            prompt = wmt16_context + '\nTranslate German to English: ' + item['translation']['de'] + '\nAnswer: '
            prompts.append(prompt)
            chat_instruction.append("Translate German text into fluent and accurate English.")

        tinystories = load_dataset("roneneldan/TinyStories", name='default', split='validation').shuffle(4242).select(range(2))
        for i in range(2):
            item = tinystories[i]
            prompt =  'Write a tiny story: ' + item['text'] + '\nWrite a tiny story:'
            prompts.append(prompt)
            chat_instruction.append("Write a coherent and engaging story based on the prompt.")

        random.seed(2024)
        random_datas = [random.randint(1, 121) for _ in range(2)]
        random_shots = [random.randint(1, 426) for _ in range(2)]
        task_path = '/home/mingbo/datasets/Spider2/'
        for i in range(2):
            file_name = f"train/{random_datas[i]}.json"
            sql_file_path = os.path.join(task_path, file_name)
            with open(sql_file_path, 'r') as f:
                item = json.load(f)
            sql_context = 'Instruction: ' + item['instruction'] + '\SQL Query: ' + item['content'].replace('\n', '')
            file_name = f"test/{random_shots[i]}.json"
            sql_file_path = os.path.join(task_path, file_name)
            with open(sql_file_path, 'r') as f:
                item = json.load(f)
            prompt = sql_context + '\nInstruction: ' + item['instruction'] + '\SQL Query:'
            prompts.append(prompt)
            chat_instruction.append("Convert natural language into correct SQL queries.")
    
    if model_id.endswith('chat-o'):
        for i in range(len(prompts)):
            messages = [
                {"role": "system", "content": chat_instruction[i]},
                {"role": "user", "content": prompts[i]},
            ]
            prompts[i] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    elif model_id.endswith('instruct'):
        for i in range(len(prompts)):
            messages = [
                {"role": "user", "content": prompts[i]},
            ]
            prompts[i] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )


    layer_searching = LayerSkippingSearching(model, tokenizer, prompts,
                                             evaluate_config={"generate_fn": "essg", "max_new_tokens": max_new_tokens})
    

    layer_searching.search(search_iter) 
    skip_attn_layers, skip_mlp_layers = layer_searching.get_solution()
    print(skip_attn_layers, skip_mlp_layers)
    with open(file_path, "w") as fout:
        fout.write("skip_attn_layers: {}\n".format(str(skip_attn_layers)))
        fout.write("skip_mlp_layers: {}\n".format(str(skip_mlp_layers)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cnndm")
    parser.add_argument("--search-iter", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--data-num", type=int, default=8)
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,)
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="llama-2-13b",

    )
    args = parser.parse_args()

    file_path = os.path.expanduser(f'./search_results/{args.model_id}_{args.task}.txt')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    import time
    start_time = time.time()
    run_search(
        model_id=args.model_id,
        model_path=args.model_path,
        task=args.task, 
        search_iter=args.search_iter, 
        max_new_tokens=args.max_new_tokens, 
        data_num=args.data_num,
        file_path = file_path,
    )
    end_time = time.time()
    print("Time:", end_time - start_time)
    with open(file_path, "a") as fout:
        fout.write("Time: {}\n".format(str(end_time - start_time)))