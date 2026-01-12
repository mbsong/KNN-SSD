import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from kneed import KneeLocator
import argparse
import os

def find_all_anchors(task_name, model_id, num_representative=10):
    """
    Find representative vectors for all tasks using KMeans clustering.
    Visualize all vectors and representative vectors using t-SNE.
    Args:
        model_id (str): The name of the model.
        num_representative (int): The number of representative vectors.
    """
    task_names = ['cnndm', 'gsm8k', 'wmt16', 'tinystories', 'sql']
    output_file = f"data/{model_id}/all_tasks_last_hidden_vector.npy"  
    output_labels_file = f"data/{model_id}/all_tasks_label.npy"

    sub_task_names = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    if task_name == 'math':
        task_names = [f"math_{sub_task_name}" for sub_task_name in sub_task_names]
        output_file = f"data/{model_id}/all_math_tasks_last_hidden_vector.npy"  
        output_labels_file = f"data/{model_id}/all_math_tasks_label.npy"

    all_last_hidden_vectors = []
    all_labels = []
    all_vectors = []
    task_sizes = []

    for task_name in task_names:
        matching_files = [file for file in os.listdir(f"data/{model_id}") if file.startswith(task_name)]
        
        if len(matching_files) > 1:
            raise ValueError(f"Multiple files found for task '{task_name}': {matching_files}")
        elif len(matching_files) == 0:
            raise FileNotFoundError(f"No file found for task '{task_name}' in data/{model_id}/")

        file_path = os.path.join(f"data/{model_id}", matching_files[0])
        task_vectors = np.load(file_path)

        # Use only the first 20% of the data for clustering to save memory
        #task_vectors = task_vectors[:task_vectors.shape[0] // 5]
        
        task_vectors = task_vectors.reshape(task_vectors.shape[0], -1)
        all_vectors.append(task_vectors)
        task_sizes.append(task_vectors.shape[0])
                
        kmeans = KMeans(n_clusters=num_representative, random_state=42, n_init=10)
        kmeans.fit(task_vectors)
        
        representative_vectors = kmeans.cluster_centers_

        labels = [task_name] * num_representative

        all_last_hidden_vectors.append(representative_vectors)
        all_labels.extend(labels)
        print(f"{task_name}'s representative anchors are done.")

    all_last_hidden_vectors = np.vstack(all_last_hidden_vectors)
    all_labels = np.array(all_labels, dtype=str)
    all_vectors = np.vstack(all_vectors)

    np.save(output_file, all_last_hidden_vectors)
    np.save(output_labels_file, all_labels)

    print(f"Last hidden vectors saved to {output_file}")
    print(f"Labels saved to {output_labels_file}")
    print(f'Anchors\' shape is {all_last_hidden_vectors.shape}, All vectors\' shape is {all_vectors.shape}')

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    all_2d = tsne.fit_transform(np.vstack([all_vectors, all_last_hidden_vectors]))
    all_vectors_2d = all_2d[:all_vectors.shape[0]]
    rep_vectors_2d = all_2d[all_vectors.shape[0]:]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    #plt.scatter(all_vectors_2d[:, 0], all_vectors_2d[:, 1], c='lightblue', s=10, alpha=0.5, label='All Vectors')
    cmap = plt.colormaps.get_cmap('tab20')
    palette = [cmap(i / max(1, (len(task_names) - 1))) for i in range(len(task_names))]

    start_idx = 0
    for i, tn in enumerate(task_names):
        end_idx = start_idx + task_sizes[i]
        plt.scatter(
            all_vectors_2d[start_idx:end_idx, 0],
            all_vectors_2d[start_idx:end_idx, 1],
            color=palette[i],
            s=10,
            alpha=0.5,
            #label=tn
        )
        start_idx = end_idx

    plt.scatter([], [], color=palette[0], s=10, marker='o', label='All Vectors')
    plt.scatter(rep_vectors_2d[:, 0], rep_vectors_2d[:, 1], c="black", s=100, marker='x', label='Representative Vectors')
    
    plt.legend(fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    save_dir = f"assets/{model_id}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/all_and_representative_vectors.pdf", format='pdf', dpi=300)
    plt.close()
    print(f"Visualization saved to {save_dir}/all_and_representative_vectors.pdf")


def find_optimal_clusters(model_id, task_name, sub_task_name, max_clusters=10):
    """
    Find the optimal number of clusters for the given task using KMeans clustering and the elbow method.
    Args:
        model_id (str): The name of the model.
        task_name (str): The name of the task (e.g., 'cnndm').
        sub_task_name (str): The name of the sub-task (if applicable).
        max_clusters (int): The maximum number of clusters to consider for the elbow method.
    """
    if task_name == 'math':
        if sub_task_name is None:
            raise ValueError("Sub-task name must be provided for the MATH dataset.")
        task_name = f"math_{sub_task_name}"
    vectors = np.load(f'data/{model_id}/{task_name}_1000_samples.npy')
    vectors = vectors.reshape(-1, vectors.shape[-1])  # Flatten the last dimension if necessary

    # Use elbow method to determine the optimal number of clusters
    sse = []  
    silhouette_values = []  
    for num_clusters in range(2, max_clusters + 1): 
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(vectors)
        sse.append(kmeans.inertia_)  
        silhouette_values.append(silhouette_score(vectors, kmeans.labels_)) 

    kneedle = KneeLocator(range(2, max_clusters + 1), sse, curve="convex", direction="decreasing")
    optimal_clusters = kneedle.knee 
    print(f"Optimal number of clusters for {task_name} (Elbow Method): {optimal_clusters}")

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), sse, marker='o', label="SSE")
    plt.axvline(x=optimal_clusters, color='r', linestyle='--', label=f"Optimal Clusters: {optimal_clusters}")
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title('Elbow Method for Optimal Clusters')
    plt.legend()
    plt.savefig('assets/elbow_method.png', dpi=300)

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_values, marker='o', label="Silhouette Score")
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Cluster Numbers')
    plt.legend()
    plt.savefig('assets/silhouetee_values.png', dpi=300)

    return optimal_clusters


def find_task_anchors(model_id, task_name, sub_task_name, optimal_clusters, num_representative=10):
    """
    Find representative vectors for a specific task using KMeans clustering.
    This function uses the optimal number of clusters determined from the elbow method.
    Args:
        model_id (str): The name of the model.
        task_name (str): The name of the task (e.g., 'cnndm').
        sub_task_name (str): The name of the sub-task (if applicable).
        optimal_clusters (int): The optimal number of clusters determined from the elbow method.
        num_representative (int): The number of representative vectors to select from each cluster.
    """
    if task_name == 'math':
        if sub_task_name is None:
            raise ValueError("Sub-task name must be provided for the MATH dataset.")
        task_name = f"math_{sub_task_name}"
    matching_files = [file for file in os.listdir(f"data/{model_id}") if file.startswith(task_name)]
    if len(matching_files) == 0:
            raise FileNotFoundError(f"No file found for task '{task_name}' in data/{model_id}/")
    file_path = os.path.join(f"data/{model_id}", matching_files[0])
    task_vectors = np.load(file_path)
    vectors = task_vectors.reshape(task_vectors.shape[0], -1)

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    cluster_centers = kmeans.cluster_centers_

    representative_vectors = []
    representative_labels = []
    representative_indices = []

    for cluster_id in range(optimal_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_vectors = vectors[cluster_indices]
        distances = pairwise_distances(cluster_vectors, [cluster_centers[cluster_id]], metric="euclidean").flatten()
        top_k_indices = cluster_indices[np.argsort(distances)[:num_representative]]
        representative_vectors.append(vectors[top_k_indices])
        representative_labels.extend([f"{task_name}-{cluster_id+1}"] * len(top_k_indices))
        representative_indices.extend(top_k_indices.tolist())

    representative_vectors = np.vstack(representative_vectors)
    representative_labels = np.array(representative_labels)

    save_dir = f"data/{model_id}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"representative_vectors_{task_name}_{optimal_clusters}.npy"), representative_vectors)
    np.save(os.path.join(save_dir, f"representative_labels_{task_name}_{optimal_clusters}.npy"), representative_labels)
    print(f"Saved representative vectors to {save_dir}/representative_vectors_{task_name}_{optimal_clusters}.npy")
    print(f"Saved labels to {save_dir}/representative_labels_{task_name}_{optimal_clusters}.npy")
    print(f"Representative indices: {representative_indices}")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    all_2d = tsne.fit_transform(np.vstack([vectors, representative_vectors]))
    vectors_2d = all_2d[:vectors.shape[0]]
    rep_vectors_2d = all_2d[vectors.shape[0]:]

    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='lightgray', s=10, alpha=0.5, label='All Vectors')
    plt.scatter(rep_vectors_2d[:, 0], rep_vectors_2d[:, 1], c='red', s=80, marker='*', label='Representative Vectors')
    plt.legend(fontsize=14)
    plt.title(f't-SNE of All and Representative Vectors for {task_name}', fontsize=16)
    plt.tight_layout()
    asset_dir = f'assets/{model_id}'
    os.makedirs(asset_dir, exist_ok=True)
    plt.savefig(os.path.join(asset_dir, f'{task_name}_representative_vectors_vis.png'), dpi=300)
    plt.close()
    print(f"Visualization saved to {asset_dir}/{task_name}_representative_vectors_vis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
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
        "--cluster-num",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    optimal_clusters = find_optimal_clusters(args.model_id, args.task_name, args.sub_task_name)
    find_task_anchors(args.model_id, args.task_name, args.sub_task_name, args.cluster_num)
    find_all_anchors(args.task_name, args.model_id, num_representative=args.cluster_num)