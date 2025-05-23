from data import preprocess, dataset
from scipy.io import loadmat
from models.models import Generator, Discriminator, Classifier, Encoder
import torch
import os
from training.training import train_models
from utils.util_functions import print_class_distribution


if __name__ == "__main__":
    # preprocess.normalization("Salinas_corrected.mat")
    # preprocess.normalization("Indian_pines_corrected.mat")
    # preprocess.gaussian_smoothing("Indian_pines_corrected.mat", 11, 1.67)
    # preprocess.gaussian_smoothing("Salinas_corrected.mat", 11, 1.67)
    # preprocess.pca("Indian_pines_corrected.mat")
    # preprocess.pca("Salinas_corrected.mat")

    file_path = os.path.join("data", "processed", "smoothed", "Indian_pines_corrected.mat")
    file_dict = loadmat(file_path)
    last_key = list(file_dict)[-1]
    data = file_dict[last_key]

    file_path = os.path.join("data", "ground_truth", "Indian_pines_gt.mat")
    file_dict = loadmat(file_path)
    last_key = list(file_dict)[-1]
    gt = file_dict[last_key]

    file_path = os.path.join("data", "processed", "pca", "Indian_pines_corrected.mat")
    file_dict = loadmat(file_path)
    last_key = list(file_dict)[-1]
    pca = file_dict[last_key]

    train, temp, test, val = dataset.get_datasets(data, gt, pca, True, 215)

    train_models(1000, 16, None, 200, "PCA", 0.0001, "cpu",
                 train, 32, 4, 2, 0.5, 200, 0.3, 10)

