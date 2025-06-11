from scipy.io import loadmat, savemat
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from utils.util_functions import get_next_batch, reparameterize
import torch
from pathlib import Path
import numpy as np


def generate(gen: Module, dataset: Dataset, num_full_passes: int, batch_size: int, mode: str, enc: Module or None,
             device: str, folder_name: str):

    Path(f"experiments/{folder_name}").mkdir(parents=True, exist_ok=True)
    all_data = []
    label_data = []

    gen.eval()
    if enc:
        enc.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    iterator = iter(dataloader)
    num_passes = 0
    while True:
        inputs, labels, pca, new_iter = get_next_batch(iterator, dataloader)
        if new_iter != iterator:
            num_passes += 1
        if num_passes == num_full_passes:
            break
        iterator = new_iter
        inputs = inputs.to(device)
        labels = labels.to(device)
        pca = pca.to(device)

        cur_bs = inputs.size(0)

        with torch.no_grad():
            if mode == "PCA":
                z = torch.randn(cur_bs, 100).to(device)
                fake_data = gen(z, labels, pca)  # (B, 1, C)

            elif mode == "VAE":
                real_data = inputs.unsqueeze(1)  # (B, 1, C)
                mu, log_s = enc(real_data)
                z = reparameterize(mu, log_s)
                fake_data = gen(z, labels, None)  # (B, 1, C)

        labels = labels.cpu().numpy()  # (B, 1)
        save_data = fake_data.cpu().squeeze(1).numpy()  # (B, C)
        all_data.append(save_data)
        label_data.append(labels)

    label_data = np.vstack(label_data)  # (total, 1)
    save_data = np.vstack(all_data)  # (total, C)
    filepath = Path(f"experiments/{folder_name}/online_data.mat")
    data_key = "data"
    label_key = "labels"
    if filepath.exists():
        file_dict = loadmat(str(filepath))
        loaded_data = file_dict[data_key]
        loaded_labels = file_dict[label_key]
        combined = np.concatenate([loaded_data, save_data], axis=0)
        combined_labels = np.concatenate([loaded_labels, label_data], axis=0)
    else:
        combined = save_data
        combined_labels = label_data

    savemat(str(filepath), {data_key: combined, label_key: combined_labels})

    gen.train()
