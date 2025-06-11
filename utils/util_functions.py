import math
import os
from pathlib import Path
from typing import Iterator
import numpy as np
import torch
from collections import Counter
import csv

from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.io import loadmat, savemat
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch import Tensor, optim
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def stratified_sample_minimum(labels: np.ndarray, min_per_class: int = 3, total_samples: int = 200) -> np.ndarray:
    """
    Returns indices with at least min_per_class samples per class,
    leftover samples are added proportionally per class to reach total_samples,
    preserving stratification.
    """
    rng = np.random.default_rng()
    labels = np.asarray(labels)
    classes, class_counts = np.unique(labels, return_counts=True)

    selected_indices = []
    leftover_indices_per_class = {}

    # Guarantee min_per_class
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        if len(cls_indices) < min_per_class:
            raise ValueError(
                f"Class {cls} has only {len(cls_indices)} samples, less than min_per_class={min_per_class}")
        selected_indices.extend(cls_indices[:min_per_class])
        leftover_indices_per_class[cls] = cls_indices[min_per_class:]

    leftover_needed = total_samples - len(selected_indices)
    if leftover_needed <= 0:
        rng.shuffle(selected_indices)
        return np.array(selected_indices)

    # Stratified sampling from leftover
    leftover_counts = {cls: len(idxs) for cls, idxs in leftover_indices_per_class.items()}
    total_leftover = sum(leftover_counts.values())

    leftover_to_pick = {}
    for cls, count in leftover_counts.items():
        leftover_to_pick[cls] = int(np.floor(count / total_leftover * leftover_needed))

    diff = leftover_needed - sum(leftover_to_pick.values())
    if diff > 0:
        fractions = {cls: (count / total_leftover * leftover_needed) - leftover_to_pick[cls] for cls, count in
                     leftover_counts.items()}
        sorted_classes = sorted(fractions, key=fractions.get, reverse=True)
        for i in range(diff):
            leftover_to_pick[sorted_classes[i]] += 1

    for cls, n_pick in leftover_to_pick.items():
        cls_leftover = leftover_indices_per_class[cls]
        rng.shuffle(cls_leftover)
        selected_indices.extend(cls_leftover[:n_pick])

    rng.shuffle(selected_indices)
    return np.array(selected_indices)


def plot_clusters(temp_dataset: Dataset, cluster_num: int):

    data = temp_dataset.data.cpu().numpy()  # (B, C)
    true_labels = temp_dataset.labels.squeeze(1).cpu().numpy()  # (B,)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca_values = PCA(n_components=2)
    data_pca = pca_values.fit_transform(data_scaled)  # (B, 2)

    cluster_algos = [
        ("kmeans", KMeans(n_clusters=cluster_num, max_iter=10000)),
        ("dbscan", DBSCAN(eps=0.466)),
        ("gmm", GaussianMixture(n_components=cluster_num, max_iter=10000)),
        ("meanshift", MeanShift(n_jobs=4, max_iter=10000))
    ]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
    axes = axes.flatten()

    cmap = plt.cm.get_cmap("tab10")

    ax = axes[0]
    ax.scatter(
        data_pca[:, 0],
        data_pca[:, 1],
        c=true_labels,
        cmap=cmap,
        s=30,
        edgecolor="none",
        alpha=0.7
    )
    ax.set_title(f"True")

    for idx, (name, algo) in enumerate(cluster_algos):
        ax = axes[idx+1]
        if name == "gmm":
            gm = algo.fit(data)
            labels = gm.predict(data)
        else:
            fitted = algo.fit(data)
            if hasattr(fitted, "labels_"):
                labels = fitted.labels_
            elif hasattr(fitted, "labels"):
                labels = fitted.labels

        if name == "DBSCAN":
            colors = []
            unique_labels = np.unique(labels)
            cluster_labels = [lab for lab in unique_labels if lab != -1]
            max_label = max(cluster_labels) if cluster_labels else 0

            for lab in labels:
                if lab == -1:
                    colors.append("k")  # noise → black
                else:
                    colors.append(cmap(float(lab) / float(max_label)))  # normalize label→[0,1]

            ax.scatter(
                data_pca[:, 0],
                data_pca[:, 1],
                c=colors,
                s=30,
                edgecolor="none",
                alpha=0.7
            )
        else:
            ax.scatter(
                data_pca[:, 0],
                data_pca[:, 1],
                c=labels,
                cmap=cmap,
                s=30,
                edgecolor="none",
                alpha=0.7
            )
        ax.set_title(f"{name}")

    plt.tight_layout()
    plt.show()


def save_models(experiment_name: str, disc: Module, disc_opt: optim, gen: Module, gen_opt: optim, cls: Module,
                cls_opt: optim, enc: Module or None, enc_opt: optim or None, epoch_num: int) -> Path:

    save_folder = Path("experiments") / experiment_name

    save_folder.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_folder / f"checkpoint_{epoch_num}.pth"

    checkpoint = {
        'experiment_id': experiment_name,
        'disc_state_dict': disc.state_dict(),
        'disc_opt_state_dict': disc_opt.state_dict(),
        'gen_state_dict': gen.state_dict(),
        'gen_opt_state_dict': gen_opt.state_dict(),
        'cls_state_dict': cls.state_dict(),
        'cls_opt_state_dict': cls_opt.state_dict(),
    }
    if enc is not None and enc_opt is not None:
        checkpoint.update({
            'enc_state_dict': enc.state_dict(),
            'enc_opt_state_dict': enc_opt.state_dict()
        })

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_models(folder_name: str, disc: Module, disc_opt: optim, gen: Module, gen_opt: optim, cls: Module,
                cls_opt: optim, enc: Module or None, enc_opt: optim or None, epoch_num: int) -> bool:

    load_folder = Path("experiments") / folder_name

    checkpoint_path = load_folder / f"checkpoint_{epoch_num}.pth"

    if not checkpoint_path.is_file():
        return False

    checkpoint = torch.load(checkpoint_path)

    disc.load_state_dict(checkpoint['disc_state_dict'])
    disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])

    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])

    cls.load_state_dict(checkpoint['cls_state_dict'])
    cls_opt.load_state_dict(checkpoint['cls_opt_state_dict'])

    if enc is not None and enc_opt is not None:
        enc.load_state_dict(checkpoint['enc_state_dict'])
        enc_opt.load_state_dict(checkpoint['enc_opt_state_dict'])

    return True


def get_next_batch(iterator: Iterator, dataloader: DataLoader):
    try:
        inputs, labels, pca = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        inputs, labels, pca = next(iterator)
    return inputs, labels, pca, iterator


def reparameterize(mu: Tensor, log_var: Tensor):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z


def write_to_csv(experiment_name: str, latent_dim: int, cls_weight: float, alpha: float, tau: int, recon_weight: float,
                 oa: float, aa: float, kappa: float, mode: str):
    path = Path("results.csv")

    file_existed = path.exists() and path.stat().st_size > 0

    header = [
        ["latent_dim", "cls_weight", "alpha", "tau", "recon_weight", "oa", "aa", "kappa", "mode", "experiment_name"]
    ]

    data = [
        [latent_dim, cls_weight, alpha, tau, recon_weight, oa, aa, kappa, mode, experiment_name]
    ]

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_existed:
            writer.writerow(header)

        writer.writerow(data)


def print_class_distribution(dataset: Dataset):
    """
    Prints the number of samples per class and their percentage in the dataset.
    """
    labels = dataset.labels
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    total = len(labels)
    counter = Counter(labels)
    print("Class distribution:")
    for cls, count in sorted(counter.items()):
        percentage = 100.0 * count / total
        print(f"  Class {cls}: {count} samples ({percentage:.2f}%)")


def load_final_data(dataset_name: str, smoothed: bool = True):
    if smoothed:
        file_path = os.path.join("data", "processed", "smoothed", f"{dataset_name}_corrected.mat")
    else:
        file_path = os.path.join("data", "processed", "normalized", f"{dataset_name}_corrected.mat")
    file_dict = loadmat(file_path)
    last_key = list(file_dict)[-1]
    data = file_dict[last_key]

    file_path = os.path.join("data", "ground_truth", f"{dataset_name}_gt.mat")
    file_dict = loadmat(file_path)
    last_key = list(file_dict)[-1]
    gt = file_dict[last_key]

    file_path = os.path.join("data", "processed", "pca", f"{dataset_name}_corrected.mat")
    file_dict = loadmat(file_path)
    last_key = list(file_dict)[-1]
    pca = file_dict[last_key]
    return data, gt, pca


def save_train_temp(experiment_name: str, train: Dataset, temp: Dataset):
    filepath = Path("experiments") / experiment_name / "train_test.mat"

    train_data = train.data.cpu().numpy()
    train_labels = train.labels.cpu().numpy()
    train_pca = train.pca.cpu().numpy()

    temp_data = temp.data.cpu().numpy()
    temp_labels = temp.labels.cpu().numpy()
    temp_pca = temp.pca.cpu().numpy()

    train_data_key = "train_data"
    train_label_key = "train_labels"
    train_pca_key = "train_pca"

    temp_data_key = "temp_data"
    temp_label_key = "temp_labels"
    temp_pca_key = "temp_pca"

    savemat(str(filepath), {train_data_key: train_data, train_label_key: train_labels, train_pca_key: train_pca,
                            temp_data_key: temp_data, temp_label_key: temp_labels, temp_pca_key: temp_pca})


def plot_ground_truth(gt_list: list[np.ndarray], cmap, norm, titles: list[str] = None, figsize_per_plot: tuple = (4, 4)):

    n = len(gt_list)

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_plot[0] * ncols,
                                      figsize_per_plot[1] * nrows),
                             squeeze=False)

    for idx, gt in enumerate(gt_list):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        ax.imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
        ax.axis("off")

        if titles is not None:
            if idx < len(titles):
                ax.set_title(titles[idx], fontsize=12)
            else:
                ax.set_title(f"Map {idx}", fontsize=12)

    for unused in range(n, nrows * ncols):
        r = unused // ncols
        c = unused % ncols
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.show()


def predict_map(model: torch.nn.Module, data: np.ndarray, device: str, ground_truth: np.ndarray) -> np.ndarray:
    model.eval()

    h, w, b = data.shape
    n = h * w

    flat = data.reshape(n, b).astype(np.float32)
    tensor = torch.from_numpy(flat).unsqueeze(1).to(device)  # (N, 1, C)

    batch_size = 512
    preds = np.zeros((n,), dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = tensor[start:end]
            logits = model(batch)
            batch_pred = torch.argmax(logits, dim=1)
            preds[start:end] = batch_pred.cpu().numpy()

    pred_map = preds.reshape(h, w)
    pred_map += 1
    pred_map[ground_truth == 0] = 0
    return pred_map


def plot_distributions(real_dataset: Dataset, fake_data: Tensor, fake_labels: Tensor):
    class_dict = {
        1: 'Alfalfa',
        2: 'Corn-notill',
        3: 'Corn-mintill',
        4: 'Corn',
        5: 'Grass-pasture',
        6: 'Grass-trees',
        7: 'Grass-pasture-mowed',
        8: 'Hay-windrowed',
        9: 'Oats',
        10: 'Soybean-notill',
        11: 'Soybean-mintill',
        12: 'Soybean-clean',
        13: 'Wheat',
        14: 'Woods',
        15: 'Building-Grass-Trees',
        16: 'Stone-Steel-Towers'
    }

    real_data = real_dataset.data.cpu().numpy()  # (B, C)
    real_labels = real_dataset.labels.squeeze(1).cpu().numpy()  # (B,)

    fake_data = fake_data.cpu().numpy()  # (B, C)
    fake_labels = fake_labels.cpu().numpy()  # (B, 1) -> (B,)

    scaler = StandardScaler()
    real_data_scaled = scaler.fit_transform(real_data)
    pca_values = PCA(n_components=2)
    real_pca = pca_values.fit_transform(real_data_scaled)  # (B, 2)

    scaler = StandardScaler()
    fake_data_scaled = scaler.fit_transform(fake_data)
    pca_values = PCA(n_components=2)
    fake_pca = pca_values.fit_transform(fake_data_scaled)  # (B, 2)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        real_pca_class = real_pca[real_labels == idx]
        fake_pca_class = fake_pca[fake_labels == idx]
        ax.scatter(
            real_pca_class[:, 0],
            real_pca_class[:, 1],
            c='red',
            s=30,
            edgecolor="none",
            alpha=0.5
        )
        ax.scatter(
            fake_pca_class[:, 0],
            fake_pca_class[:, 1],
            c='green',
            s=30,
            edgecolor="none",
            alpha=0.5
        )
        ax.set_title(f"Class {class_dict[idx+1]}")

    plt.tight_layout()
    plt.show()
