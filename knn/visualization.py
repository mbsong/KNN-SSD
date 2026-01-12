import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap
import glob
import os

def KNN_tsne(vectors, save_path, task_name, data_num, cluster_num=3):
    tsne = TSNE(n_components=2, perplexity=20, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)

    kmeans = KMeans(n_clusters=cluster_num, random_state=42).fit(vectors)
    labels = kmeans.labels_

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.title(f't-SNE Visualization of {data_num} {task_name} last hidden vectors')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(save_path, dpi=300)


def KNN_umap(vectors,  save_path, task_name, data_num, cluster_num=3):
    umap_model = umap.UMAP(n_components=3, random_state=42)
    vectors_3d = umap_model.fit_transform(vectors)

    kmeans = KMeans(n_clusters=cluster_num, random_state=42).fit(vectors)
    labels = kmeans.labels_

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2], c=labels, cmap="tab10", alpha=0.7)

    ax.set_title(f'UMAP 3D Visualization of {data_num} {task_name} last hidden vectors')
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--data-num",
        type=int,
        default=10,
        help="The number of samples.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default='tsne',
        help="tsne or umap"
    )
    parser.add_argument(
        "--cluster-num",
        type=int,
        default=4
    )

    args = parser.parse_args()

    if args.task_name == 'math':
        npy_files = glob.glob("math*.npy")
        vectors_list = [np.load(npy_file) for npy_file in npy_files]
        vectors = np.concatenate(vectors_list, axis=0)
        vectors2 = np.load('cnndm_100_samples.npy')
        vectors = np.vstack([vectors, vectors2])
        vectors3 = np.load('gsm8k_100_samples.npy')
        vectors = np.vstack([vectors, vectors3])
    elif args.task_name == 'all':
        vectors = np.load(f'data/{args.model_id}/cnndm_1000_samples.npy')
        vectors2 = np.load(f'data/{args.model_id}/gsm8k_1000_samples.npy')
        vectors = np.vstack([vectors, vectors2])
        vectors3 = np.load(f'data/{args.model_id}/wmt16_1000_samples.npy')
        vectors = np.vstack([vectors, vectors3])
        vectors4 = np.load(f'data/{args.model_id}/tinystories_1000_samples.npy')
        vectors = np.vstack([vectors, vectors4])
        vectors5 = np.load(f'data/{args.model_id}/sql_1000_samples.npy')
        vectors = np.vstack([vectors, vectors5])
        save_path=f'assets/KNN_all.png'
    else:   
        vectors = np.load(f'data/{args.model_id}/{args.task_name}_{args.data_num}_samples.npy')
        save_path=f'assets/{args.model_id}/KNN_{args.task_name}_{args.data_num}_{args.cluster_num}_{args.method}.png'

    print(f'Output to {save_path}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f'Vectors\' shape is {vectors.shape}')
    if args.method == 'tsne':
        KNN_tsne(vectors, 
                 save_path,
                 task_name=args.task_name, 
                 data_num=args.data_num,
                 cluster_num=args.cluster_num)
    else:
        KNN_umap(vectors,
                 save_path,
                 task_name=args.task_name, 
                 data_num=args.data_num,
                 cluster_num=args.cluster_num)

    print(f'Results have been generated')

