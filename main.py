from pathlib import Path
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from data import preprocess, dataset
from scipy.io import loadmat
import torch.nn.functional as F
from data.KNN_selection import cluster_set, knn_select
from data.dataset import HyperspectralDataset
from evaluation.eval import evaluate, evaluate_with_val
from models.models import Generator, Discriminator, Classifier, Encoder, CNN1D, CNN3D
import torch
from torch.utils.data import DataLoader
import os
from training.training import train_models, train_cls
from utils.util_functions import print_class_distribution, load_final_data, write_to_csv, plot_clusters, \
    save_train_temp, plot_ground_truth, predict_map, plot_distributions, reparameterize
import random
import csv


if __name__ == "__main__":
    # preprocess.normalization("Salinas_corrected.mat")
    # preprocess.normalization("Indian_pines_corrected.mat")
    # preprocess.gaussian_smoothing("Indian_pines_corrected.mat", 11, 1.67)
    # preprocess.gaussian_smoothing("Salinas_corrected.mat", 11, 1.67)
    # preprocess.pca("Indian_pines_corrected.mat")
    # preprocess.pca("Salinas_corrected.mat")

    data, gt, pca = load_final_data("Salinas")
    train_size = 543

    # AC-WGAN-GP training
    '''
    train, temp = dataset.get_datasets(data, gt, pca, False, train_size)

    mode = "VAE"

    experiment_name = train_models(5001, 16, 60, 204, mode,
                                   1e-4, 1e-3, "cuda", train, 64, 1, 1,
                                   0.3, 2500, 1, 10, 1000,
                                   10, 1000, None, None, mode, True)

    save_train_temp(experiment_name, train, temp)
    '''

    # Classification
    '''
    oa_avg, aa_avg, kappa_avg = 0, 0, 0

    val_iters = 10

    for _ in range(val_iters):
        train, test = dataset.get_datasets_as_patches(data, gt, pca, True, train_size)

        cls = CNN3D(16, 204).to('cuda')

        oa, aa, kappa = evaluate(train, test, 64, cls, 1e-4, 30000, 16, 'cuda')

        oa_avg += oa
        aa_avg += aa
        kappa_avg += kappa

    oa_avg /= val_iters
    aa_avg /= val_iters
    kappa_avg /= val_iters
    print(oa_avg, aa_avg, kappa_avg)
    '''

    # Classification (augmented)

    val_iters = 10
    experiment_name = 'ca3c4w47'
    ratios = [1, 2, 3, 4]
    results = {}

    _, temp = dataset.get_datasets(data, gt, pca, True, train_size)

    result_real = cluster_set(temp, 120, 16, 2.5)

    for ratio in ratios:

        oa_avg, aa_avg, kappa_avg = 0, 0, 0

        fake_data, labels_smoothed = knn_select(result_real, experiment_name, ratio, 16)  # (N, C) and (N, class_num)

        for _ in range(val_iters):
            train, test = dataset.get_datasets(data, gt, pca, True, train_size)

            cls = Classifier(16).to('cuda')

            train_labels = F.one_hot(train.labels.squeeze(1), num_classes=16).cpu()  # (N, class_num)

            train_data = torch.cat([train.data, fake_data], 0)  # (N, C)
            train_labels = torch.cat([train_labels, labels_smoothed], dim=0)
            train_pca = torch.empty(train_data.size(dim=0))

            train = HyperspectralDataset(train_data, train_labels, train_pca)

            oa, aa, kappa = evaluate(train, test, 64, cls, 1e-4, 30000, 16, 'cuda')

            oa_avg += oa
            aa_avg += aa
            kappa_avg += kappa

        oa_avg /= val_iters
        aa_avg /= val_iters
        kappa_avg /= val_iters

        results[ratio] = [oa_avg, aa_avg, kappa_avg]

    print(results)

    # Segmentation
    '''
    fixed_color_list = [
        "#000000",  # 0: background → black
        "#e6194b",  # 1: red
        "#3cb44b",  # 2: green
        "#ffe119",  # 3: yellow
        "#4363d8",  # 4: blue
        "#f58231",  # 5: orange
        "#911eb4",  # 6: purple
        "#46f0f0",  # 7: cyan
        "#f032e6",  # 8: magenta
        "#bcf60c",  # 9: lime
        "#fabebe",  # 10: pink
        "#008080",  # 11: teal
        "#e6beff",  # 12: lavender
        "#9a6324",  # 13: brown
        "#fffac8",  # 14: beige
        "#800000",  # 15: maroon
        "#aaffc3"  # 16: mint
    ]
    cmap = ListedColormap(fixed_color_list)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, 17.5), ncolors=17)

    ground_truths = []
    device = 'cuda'
    _, gt, _ = load_final_data(smoothed=True)

    ground_truths.append(gt)

    # 1D-CNN
    data, gt, pca = load_final_data(smoothed=False)
    train, _ = dataset.get_datasets(data, gt, pca, True)
    cls = CNN1D().to(device)
    cls = train_cls(train, 64, cls, 1e-4, 20000, device)
    ground_truth = predict_map(cls, data, device, gt)
    ground_truths.append(ground_truth)

    # 1D-S-CNN
    data, gt, pca = load_final_data(smoothed=True)
    experiment_name = '6s7kbg5k'
    ratio = 3
    train, test = dataset.get_datasets(data, gt, pca, True)

    result_real = cluster_set(test, 38, 16, 2.5)
    fake_data, labels_smoothed = knn_select(result_real, experiment_name, ratio, 16)

    train_labels = F.one_hot(train.labels.squeeze(1), num_classes=16).cpu()
    train_data = torch.cat([train.data, fake_data], 0)
    train_labels = torch.cat([train_labels, labels_smoothed], dim=0)
    train_pca = torch.empty(train_data.size(dim=0))
    train_aug = HyperspectralDataset(train_data, train_labels, train_pca)

    cls = Classifier(16).to(device)
    cls = train_cls(train, 64, cls, 1e-4, 20000, device)
    ground_truth = predict_map(cls, data, device, gt)
    ground_truths.append(ground_truth)

    cls = Classifier(16).to(device)
    cls = train_cls(train_aug, 64, cls, 1e-4, 20000, device)
    ground_truth = predict_map(cls, data, device, gt)
    ground_truths.append(ground_truth)

    plot_ground_truth(ground_truths, cmap, norm, titles=["Ground Truth", "1D-CNN", "1D-S-CNN", "AC-WGAN-GP"])
    '''

    # Distribution
    '''
    data, gt, pca = load_final_data(smoothed=True)
    experiment_name = '6s7kbg5k'
    ratio = 10
    device = 'cuda'
    train, temp = dataset.get_datasets(data, gt, pca, True)

    load_folder = Path("experiments") / experiment_name
    checkpoint_path = load_folder / f"checkpoint_5000.pth"
    checkpoint = torch.load(checkpoint_path)
    
    result_real = cluster_set(temp, 38, 16, 2.5)
    fake_data, labels_smoothed = knn_select(result_real, experiment_name, ratio, 16)
    fake_labels = torch.argmax(labels_smoothed, dim=1)

    plot_distributions(temp, fake_data, fake_labels)
    '''
