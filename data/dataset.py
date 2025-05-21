import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils.util_functions import stratified_sample_minimum


class HyperspectralDataset(Dataset):
    def __init__(self, data, labels, pca):
        self.data = data
        self.labels = labels
        self.pca = pca

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        pca = self.pca[idx] if self.pca is not None else None
        return x, y, pca


def get_datasets(dataset: np.ndarray, ground_truth: np.ndarray, pca: np.ndarray, valid: bool = False,
                 train_size: int = 215, val_size: int = 1000, random_state=10) -> tuple[HyperspectralDataset, ...]:
    # reshape the data into (N, C) or (N, 1)
    w, h, c = dataset.shape
    dataset = dataset.reshape((w * h, -1))
    ground_truth = ground_truth.flatten()

    # remove all background pixels
    mask = ground_truth != 0
    dataset = dataset[mask]
    pca = pca[mask]
    ground_truth = ground_truth[mask]

    train_idx = stratified_sample_minimum(
        labels=ground_truth,
        min_per_class=3,
        total_samples=train_size,
        random_state=random_state
    )
    temp_idx = np.setdiff1d(np.arange(len(dataset)), train_idx)

    dataset_train = torch.from_numpy(dataset[train_idx].astype(np.float32))
    labels_train = torch.from_numpy(ground_truth[train_idx].astype(np.int64))
    pca_train = torch.from_numpy(pca[train_idx].astype(np.float32))

    train_dataset = HyperspectralDataset(dataset_train, labels_train, pca_train)

    dataset_temp = dataset[temp_idx]
    labels_temp = ground_truth[temp_idx]

    dataset_temp_tensor = torch.from_numpy(dataset_temp.astype(np.float32))
    labels_temp_tensor = torch.from_numpy(labels_temp.astype(np.int64))

    if valid:
        '''
        valid_idx, test_idx = train_test_split(
            np.arange(len(dataset_temp)),
            stratify=labels_temp,
            train_size=val_size,
            random_state=random_state,
        )
        '''

        valid_idx = stratified_sample_minimum(
            labels=labels_temp,
            min_per_class=3,
            total_samples=val_size,
            random_state=random_state
        )
        test_idx = np.setdiff1d(np.arange(len(dataset_temp)), valid_idx)

        dataset_val = dataset_temp_tensor[valid_idx]
        labels_val = labels_temp_tensor[valid_idx]

        dataset_test = dataset_temp_tensor[test_idx]
        labels_test = labels_temp_tensor[test_idx]

        temp_dataset = HyperspectralDataset(dataset_temp_tensor, labels_temp_tensor, None)
        test_dataset = HyperspectralDataset(dataset_test, labels_test, None)
        val_dataset = HyperspectralDataset(dataset_val, labels_val, None)

        return train_dataset, temp_dataset, test_dataset, val_dataset
    else:
        dataset_test = dataset_temp_tensor
        labels_test = labels_temp_tensor

        temp_dataset = HyperspectralDataset(dataset_temp_tensor, labels_temp_tensor, None)
        test_dataset = HyperspectralDataset(dataset_test, labels_test, None)

        return train_dataset, temp_dataset, test_dataset
