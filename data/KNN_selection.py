from pathlib import Path
import torch
import torch.nn.functional as F
from numpy import random
from scipy.io import loadmat
from sklearn.cluster import KMeans, DBSCAN, mean_shift
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from data.dataset import HyperspectralDataset
import numpy as np


def cluster_set(temp_dataset: HyperspectralDataset, m: int, cluster_num: int, radius_factor: float) -> np.ndarray:

    data = temp_dataset.data  # tensor (B, C)

    data = data.cpu().numpy()

    cluster_algos = [
        ("kmeans", KMeans(n_clusters=cluster_num, max_iter=10000)),
        ("dbscan", DBSCAN(eps=0.19)),  # 0.466 - Indian Pines
        ("gmm", GaussianMixture(n_components=cluster_num, max_iter=10000)),
        ("meanshift", None)
    ]

    centers_set = []
    random_data = []

    for name, algo in cluster_algos:
        if name == "gmm":
            gm = algo.fit(data)
            centers = gm.means_
            print(f"gmm centers shape: {centers.shape}")
            print(f"gmm convergence: {gm.converged_}")
        elif name == "dbscan":
            db = algo.fit(data)
            centers = []
            for lbl in set(db.labels_) - {-1}:
                pts = data[db.labels_ == lbl]
                centers.append(pts.mean(axis=0))
            centers = np.vstack(centers) if centers else np.empty((0, data.shape[1]))
            print(f"dbscan centers shape: {centers.shape}")
        elif name == "meanshift":
            centers, _ = mean_shift(data, bandwidth=0.6, n_jobs=-1, max_iter=10000)
            print(f"{name} centers shape: {centers.shape}")
        else:
            centers = algo.fit(data).cluster_centers_
            print(f"{algo} centers shape: {centers.shape}")

        candidates = []

        for c in centers:
            dist = np.linalg.norm(data - c, axis=1)
            idx = np.argmin(dist)
            center = data[idx]
            centers_set.append(center)
            r = radius_factor * np.median(dist)
            near_mask = (dist > 0) & (dist <= r)
            candidates.extend(np.flatnonzero(near_mask))

        candidates = np.unique(candidates)

        if len(candidates) < m:
            chosen_idx = random.choice(candidates, size=m, replace=True)
        else:
            chosen_idx = random.choice(candidates, size=m, replace=False)

        chosen_data = data[chosen_idx]  # (m, C)
        random_data.append(chosen_data)

    centers = np.vstack(centers_set)  # (4N, C)
    random_data = np.vstack(random_data)  # (4M, C)

    result_set = np.vstack((centers, random_data))  # (4(N+M), C)
    print(f"result set shape: {result_set.shape}")

    return result_set


def knn_select(real_data: np.ndarray, experiment_name: str, k: int, class_num: int, eps: float = 0.1):
    gen_path = Path(f"experiments/{experiment_name}/online_data.mat")
    file_dict = loadmat(str(gen_path))
    gen_data = file_dict["data"]  # (N, C)
    gen_labels = file_dict["labels"].ravel()  # (N,)

    nn = NearestNeighbors(n_neighbors=k)
    nn = nn.fit(gen_data)
    _, idxs = nn.kneighbors(real_data)
    flat_idxs = idxs.ravel()

    gen_data = torch.as_tensor(gen_data[flat_idxs], dtype=torch.float32)  # (N, C)
    gen_labels = torch.as_tensor(gen_labels[flat_idxs], dtype=torch.long)  # (N,)

    # LABEL SMOOTHING

    one_hot = F.one_hot(gen_labels, num_classes=class_num).float()
    labels_smoothed = one_hot * (1.0 - eps) + eps / class_num

    return gen_data, labels_smoothed
