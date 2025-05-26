from pathlib import Path
from typing import Iterator
import numpy as np
import torch
from collections import Counter
from torch import Tensor, optim
from torch.nn import Module
from torch.optim import lr_scheduler
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


def save_models(experiment_name: str, disc: Module, disc_opt: optim, disc_sch: lr_scheduler, gen: Module,
                gen_opt: optim, gen_sch: lr_scheduler, cls: Module, cls_opt: optim, enc: Module or None,
                enc_opt: optim or None, enc_sch: lr_scheduler or None, epoch_num: int) -> Path:

    save_folder = Path("experiments") / experiment_name

    save_folder.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_folder / f"checkpoint_{epoch_num}.pth"

    checkpoint = {
        'experiment_id': experiment_name,
        'disc_state_dict': disc.state_dict(),
        'disc_opt_state_dict': disc_opt.state_dict(),
        'disc_sch_state_dict': disc_sch.state_dict(),
        'gen_state_dict': gen.state_dict(),
        'gen_opt_state_dict': gen_opt.state_dict(),
        'gen_sch_state_dict': gen_sch.state_dict(),
        'cls_state_dict': cls.state_dict(),
        'cls_opt_state_dict': cls_opt.state_dict(),
    }
    if enc is not None and enc_opt is not None and enc_sch is not None:
        checkpoint.update({
            'enc_state_dict': enc.state_dict(),
            'enc_opt_state_dict': enc_opt.state_dict(),
            'enc_sch_state_dict': enc_sch.state_dict(),
        })

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_models(folder_name: str, disc: Module, disc_opt: optim, disc_sch: lr_scheduler, gen: Module, gen_opt: optim,
                gen_sch: lr_scheduler, cls: Module, cls_opt: optim, enc: Module or None, enc_opt: optim or None,
                enc_sch: lr_scheduler or None, epoch_num: int) -> bool:

    load_folder = Path("experiments") / folder_name

    checkpoint_path = load_folder / f"checkpoint_{epoch_num}.pth"

    if not checkpoint_path.is_file():
        return False

    checkpoint = torch.load(checkpoint_path)

    disc.load_state_dict(checkpoint['disc_state_dict'])
    disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])
    disc_sch.load_state_dict(checkpoint['disc_sch_state_dict'])

    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])
    gen_sch.load_state_dict(checkpoint['gen_sch_state_dict'])

    cls.load_state_dict(checkpoint['cls_state_dict'])
    cls_opt.load_state_dict(checkpoint['cls_opt_state_dict'])

    if enc is not None and enc_opt is not None and enc_sch is not None:
        enc.load_state_dict(checkpoint['enc_state_dict'])
        enc_opt.load_state_dict(checkpoint['enc_opt_state_dict'])
        enc_sch.load_state_dict(checkpoint['enc_sch_state_dict'])

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