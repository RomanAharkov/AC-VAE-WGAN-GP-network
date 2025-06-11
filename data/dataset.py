import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils.util_functions import stratified_sample_minimum


class HyperspectralDataset(Dataset):
    def __init__(self, data, labels, pca):
        self.data = data
        self.labels = labels  # (N, 1)
        self.pca = pca

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        pca = self.pca[idx]
        return x, y, pca


def get_datasets(dataset: np.ndarray, ground_truth: np.ndarray, pca: np.ndarray, stratified: bool,
                 train_size: int = 215) -> tuple[HyperspectralDataset, ...]:
    # reshape the data into (N, C) or (N, 1)
    w, h, c = dataset.shape
    dataset = dataset.reshape((w * h, -1))
    ground_truth = ground_truth.flatten()

    # remove all background pixels
    mask = ground_truth != 0
    dataset = dataset[mask]
    pca = pca[mask]
    ground_truth = ground_truth[mask]

    # change the ground_truth to one lower
    ground_truth -= 1

    if stratified:
        train_idx = stratified_sample_minimum(
            labels=ground_truth,
            min_per_class=3,
            total_samples=train_size
        )
    else:
        train_idx = np.random.choice(len(dataset), size=train_size, replace=False)

    test_idx = np.setdiff1d(np.arange(len(dataset)), train_idx)

    dataset_train = torch.tensor(dataset[train_idx], dtype=torch.float32)
    labels_train = torch.tensor(ground_truth[train_idx], dtype=torch.long).unsqueeze(1)
    pca_train = torch.tensor(pca[train_idx], dtype=torch.float32)

    train_dataset = HyperspectralDataset(dataset_train, labels_train, pca_train)

    dataset_test = dataset[test_idx]
    labels_test = ground_truth[test_idx]
    pca_test = pca[test_idx]

    dataset_test_tensor = torch.tensor(dataset_test, dtype=torch.float32)
    labels_test_tensor = torch.tensor(labels_test, dtype=torch.long)
    pca_test_tensor = torch.tensor(pca_test, dtype=torch.float32)

    test_dataset = HyperspectralDataset(dataset_test_tensor, labels_test_tensor.unsqueeze(1), pca_test_tensor)

    return train_dataset, test_dataset


def get_datasets_as_patches(dataset: np.ndarray, ground_truth: np.ndarray, pca: np.ndarray, stratified: bool,
                            train_size: int = 215, patch_size: int = 5) -> tuple[Dataset, Dataset]:

    h, w, c = dataset.shape
    pad = patch_size // 2   # e.g. 5//2 = 2

    padded_cube = np.pad(dataset, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    gt_flat = ground_truth.flatten()

    coords_all = np.argwhere(ground_truth != 0)

    gt_zero_based = gt_flat.astype(np.int64) - 1

    valid_indices_flat = (coords_all[:, 0] * w + coords_all[:, 1]).astype(np.int64)

    if stratified:
        strat_idx = stratified_sample_minimum(
            labels=gt_zero_based[valid_indices_flat],
            min_per_class=3,
            total_samples=train_size
        )
        train_flat_indices = valid_indices_flat[strat_idx]
    else:
        rng = np.random.default_rng()
        choice = rng.choice(len(valid_indices_flat), size=train_size, replace=False)
        train_flat_indices = valid_indices_flat[choice]

    test_flat_indices = np.setdiff1d(valid_indices_flat, train_flat_indices)

    def extract_patch(flat_idx: int) -> np.ndarray:
        i = flat_idx // w
        j = flat_idx % w
        i_pad = i + pad
        j_pad = j + pad

        patch = padded_cube[ (i_pad - pad):(i_pad + pad + 1), (j_pad - pad):(j_pad + pad + 1), :]
        return patch  # (patch_size, patch_size, C)

    train_patches = []
    train_labels = []
    train_pcas = []

    for flat_idx in train_flat_indices:
        patch = extract_patch(flat_idx)  # (patch_size, patch_size, C)
        train_patches.append(patch)
        train_labels.append(int(gt_zero_based[flat_idx]))
        train_pcas.append(pca[flat_idx])

    train_patches = np.stack(train_patches, axis=0)  # shape = (train_size, patch_size, patch_size, C)
    train_labels = np.array(train_labels, dtype=np.int64)  # (train_size,)
    train_pcas = np.stack(train_pcas, axis=0)

    test_patches = []
    test_labels = []
    test_pcas = []

    for flat_idx in test_flat_indices:
        patch = extract_patch(flat_idx)
        test_patches.append(patch)
        test_labels.append(int(gt_zero_based[flat_idx]))
        test_pcas.append(pca[flat_idx])

    test_patches = np.stack(test_patches, axis=0)  # (N_test, patch_size, patch_size, C)
    test_labels = np.array(test_labels, dtype=np.int64)  # (N_test,)
    test_pcas = np.stack(test_pcas, axis=0)

    train_patches_tensor = torch.from_numpy(train_patches).float().permute(0, 3, 1, 2).unsqueeze(1)
    # (train_size, 1, C, patch_size, patch_size)

    train_labels_tensor = torch.from_numpy(train_labels).long()
    train_pcas_tensor = torch.from_numpy(train_pcas).float()

    test_patches_tensor = torch.from_numpy(test_patches).float().permute(0, 3, 1, 2).unsqueeze(1)
    test_labels_tensor = torch.from_numpy(test_labels).long()
    test_pcas_tensor = torch.from_numpy(test_pcas).float()

    train_dataset = HyperspectralDataset(train_patches_tensor, train_labels_tensor.unsqueeze(1), train_pcas_tensor)
    test_dataset = HyperspectralDataset(test_patches_tensor,  test_labels_tensor.unsqueeze(1),  test_pcas_tensor)

    return train_dataset, test_dataset
