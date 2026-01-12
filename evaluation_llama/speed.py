import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def speed(jsonl_file, jsonl_file_base, report=True, report_sample=True):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if 'choices' in json_obj:
                data.append(json_obj)
    data_num = len(data)
    speeds=[]
    accept_lengths_list = []
    accept_rate = 0
    for datapoint in data[:data_num]:
        tokens = sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        accept_lengths_list.extend(datapoint["choices"][0]['accept_lengths'])
        accept_rate += datapoint["choices"][0]['acceptance_rate']
        speeds.append(tokens/times)


    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if 'choices' in json_obj:
                data.append(json_obj)

    if len(data) != data_num:
        print(f"Warning: The number of samples in {jsonl_file} and {jsonl_file_base} do not match.")

    speeds0=[]
    for datapoint in data[:data_num]:
        tokens = sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens/times)

    tokens_per_second = np.array(speeds).mean()
    tokens_per_second_baseline = np.array(speeds0).mean()
    speedup_ratio = np.array(speeds).mean()/np.array(speeds0).mean()
    accept_rate = accept_rate / data_num

    sample_speedup = []
    avg_speedup = []
    
    if report_sample:
        for i in range(data_num):
            print("Tokens per second: ", speeds[i])
            print("Tokens per second for the baseline: ", speeds0[i])
            sample_speedup.append(speeds[i]/speeds0[i])
            print("Sample Speedup: ", sample_speedup[i])
            avg_speedup.append(np.array(speeds[:i+1]).mean()/np.array(speeds0[:i+1]).mean())
            print(f"Avg Speedup: {avg_speedup[i]}\n")

    if report:
        print("="*30, "Overall: ", "="*30)
        print("#Mean accepted tokens: ", np.mean(accept_lengths_list))
        print("#Mean acceptance rate: ", accept_rate)
        print('Tokens per second: ', tokens_per_second)
        print('Tokens per second for the baseline: ', tokens_per_second_baseline)
        print("Speedup ratio: ", speedup_ratio)
    return sample_speedup, avg_speedup, data_num

def plot_speedup(sample_speedup, speedup, data_num, model_id, seed=None):
        
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(range(1, data_num + 1), sample_speedup, color='skyblue', alpha=0.7, label='Sample Speedup')
    ax1.set_xlabel('Sample Index', size=16)
    ax1.set_ylabel('Sample Speedup', color='blue', size=16)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(range(1, data_num + 1), speedup, color='orange', marker='o', label='Average Speedup')
    ax2.set_ylabel('Average Speedup', color='orange', size=16)
    ax2.tick_params(axis='y', labelcolor='orange')

    ax2.axhline(y=speedup[-1], color='green',linestyle='--', linewidth=1.5)

    plt.title('Speedup Visualization')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    fig.legend(loc='lower right', bbox_to_anchor=(0.9, 0.1))
    os.makedirs(os.path.dirname(f'../assets/{model_id}'), exist_ok=True)
    if seed:
        plt.savefig(f'../assets/{model_id}/speedup_visualization_{seed}.pdf', format='pdf', dpi=300)
    else:
        plt.savefig(f'../assets/{model_id}/speedup_visualization_Overall.pdf', format='pdf', dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_id = 'llama-2-13b'
    dataset = 'knn'
    data_per_task = 40
    task_num = 5
    seed = 2026
    mix_ratio = 0.0
    parser.add_argument(
        "--file-path",
        default=f'../results/knn/{dataset}_{data_per_task}_{mix_ratio}/{model_id}/knn-seed-{seed}.jsonl',
        type=str,
        help="The file path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--file-folder",
        default=None,
        type=str,
        help="The file folder path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--base-path",
        default=f'../results/baseline/mixed_{data_per_task}_{mix_ratio}/{model_id}/vanilla-seed-{seed}.jsonl',
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--base-folder",
        default=None,
        type=str,
        help="The file folder path of evaluated baseline.",
    )

    args = parser.parse_args()

    if args.file_folder and args.base_folder:

        total_sample_speedup = np.zeros(data_per_task * task_num)
        total_avg_speedup = np.zeros(data_per_task * task_num)
        file_count = 0
        seeds = [2024, 2025, 6859, 24536, 723069, 934857]

        for seed in seeds:
            file_path = os.path.join(args.file_folder, f'knn-seed-{seed}.jsonl')
            base_path = os.path.join(args.base_folder, f'vanilla-seed-{seed}.jsonl')
            print(f"Processing file: {file_path} and {base_path}")
            sample_speedup, avg_speedup, data_num = speed(jsonl_file=file_path, jsonl_file_base=base_path)
            plot_speedup(sample_speedup, avg_speedup, data_num, model_id=model_id, seed=seed)
                
            total_sample_speedup += np.array(sample_speedup)
            total_avg_speedup += np.array(avg_speedup)
            file_count += 1
            
        if file_count > 0:
            mean_sample_speedup = total_sample_speedup / file_count
            mean_avg_speedup = total_avg_speedup / file_count

            plot_speedup(mean_sample_speedup, mean_avg_speedup, data_num, 'Overall')
            print("\n", "="*25, f"Overall for {file_count} files: ", "="*25)
            print(f"Mean Average Speedup: {mean_avg_speedup[data_num-1]}")
    else:
        sample_speedup, avg_speedup, data_num = speed(jsonl_file=args.file_path, jsonl_file_base=args.base_path)
        plot_speedup(sample_speedup, avg_speedup, data_num, model_id=model_id, seed=seed)
        print(f"Plot saved as speedup_visualization.pdf")