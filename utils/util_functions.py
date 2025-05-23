from typing import Iterator
import numpy as np
import torch
from collections import Counter
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


def stratified_sample_minimum(labels: np.ndarray, min_per_class: int = 3, total_samples: int = 200, random_state=10) \
        -> np.ndarray:
    """
    Returns indices with at least min_per_class samples per class,
    leftover samples are added proportionally per class to reach total_samples,
    preserving stratification.
    """
    rng = np.random.default_rng(seed=random_state)
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