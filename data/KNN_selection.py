from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from data.dataset import HyperspectralDataset
import numpy as np


def cluster_set(temp_dataset: HyperspectralDataset, ratio: float, train_size: int, cluster_num: int) -> (
        tuple[np.ndarray, int]):
    # ratio = fake/real
    if ratio >= 1:
        k = int(ratio)
        cluster_size = ceil(train_size / 4 - cluster_num)
    elif ratio < 1:
        k = 1
        cluster_size = ceil(train_size * ratio / 4 - cluster_num)

    data = temp_dataset.data  # tensor (B, C)

    data = data.cpu().numpy()

    cluster_algos = [
        ("kmeans", KMeans(n_clusters=cluster_num, random_state=0)),
        ("dbscan", DBSCAN()),
        ("gmm", GaussianMixture(n_components=cluster_num, random_state=0)),
        ("meanshift", MeanShift())
    ]

    result_set = []

    for name, algo in cluster_algos:
        if name == "gmm":
            labels = algo.fit_predict(data)
            centers = algo.means_
        elif name == "dbscan":
            labels = algo.fit_predict(data)
            # ignore noise points labeled -1
            centers = []
            for lbl in set(labels) - {-1}:
                pts = data[labels == lbl]
                centers.append(pts.mean(axis=0))
            centers = np.vstack(centers) if centers else np.empty((0, data.shape[1]))
        else:
            centers = algo.fit(data).cluster_centers_

        for c in centers:
            idx = np.argmin(np.linalg.norm(data - c, axis=1))
            result_set.append(data[idx])

        rand_idx = np.random.choice(len(data), size=cluster_size, replace=False)
        result_set.extend(data[rand_idx])

    result_set = np.vstack(result_set)  # (4(N+M), C)

    return result_set, k


def knn_select(real_data: np.ndarray, experiment_name: str, k: int, fake_size: int, class_num: int, eps: float = 0.1):
    gen_path = Path(f"experiments/{experiment_name}/online/online_data.mat")
    file_dict = loadmat(str(gen_path))
    gen_data = file_dict["data"]  # (N, C)
    gen_labels = file_dict["labels"]  # (N, 1)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(gen_data)
    _, idxs = nn.kneighbors(real_data)
    flat_idxs = idxs.flatten()

    unique_idxs = np.unique(flat_idxs)
    if fake_size > len(unique_idxs):
        extra = np.random.choice(flat_idxs, size=(fake_size - len(unique_idxs)), replace=True)
        chosen = np.concatenate([unique_idxs, extra])
    else:
        chosen = unique_idxs

    gen_data = torch.tensor(gen_data[chosen], dtype=torch.float32)  # (N, C)
    gen_labels = torch.tensor(gen_labels[chosen].flatten()).long()  # (N,)

    # LABEL SMOOTHING
    gen_labels = F.one_hot(gen_labels, num_classes=class_num)

    labels_smoothed = gen_labels * (1 - eps) + eps / class_num

    return gen_data, labels_smoothed
