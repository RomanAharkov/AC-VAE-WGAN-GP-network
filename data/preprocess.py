import os
import cv2
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import convolve2d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def gaussian_kernel(size=11, sigma=1.67):
    k = cv2.getGaussianKernel(size, sigma)
    kernel = np.outer(k, k)
    return kernel


def spatial_convolve_hsi(hsi: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply 2D spatial convolution to each spectral band in HSI.

    Args:
        hsi: numpy array of shape [H, W, C]
        kernel: 2D convolution kernel

    Returns:
        Smoothed HSI of shape [H, W, C]
    """
    h, w, c = hsi.shape
    smoothed = np.zeros_like(hsi)

    for i in range(c):
        smoothed[:, :, i] = convolve2d(hsi[:, :, i], kernel, mode='same', boundary='symm')

    return smoothed


def gaussian_smoothing(filename: str, window_size: int, sigma: float) -> None:
    file_path = os.path.join("data", "processed", "normalized", filename)
    file_dict = loadmat(file_path)
    last_key = list(file_dict)[-1]
    data = file_dict[last_key]

    kernel = gaussian_kernel(window_size, sigma)
    smoothed_data = spatial_convolve_hsi(data, kernel)

    save_path = os.path.join("data", "processed", "smoothed", filename)
    savemat(save_path, {last_key: smoothed_data})


def normalization(filename: str) -> None:
    file_path = os.path.join("data", "raw", filename)
    file_dict = loadmat(file_path)
    last_key = list(file_dict)[-1]
    data = file_dict[last_key]

    min_vals = data.min((0, 1))
    max_vals = data.max((0, 1))
    data = (data - min_vals) / (max_vals - min_vals)
    data = 2 * data - 1

    save_path = os.path.join("data", "processed", "normalized", filename)
    savemat(save_path, {last_key: data})


def pca(filename: str, components: int = 30) -> None:
    file_path = os.path.join("data", "processed", "smoothed", filename)
    file_dict = loadmat(file_path)
    last_key = list(file_dict)[-1]
    data = file_dict[last_key]

    _, _, c = data.shape
    data_reshaped = data.reshape(-1, c)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    pca_values = PCA(n_components=components)
    data_pca = pca_values.fit_transform(data_scaled)

    save_path = os.path.join("data", "processed", "pca", filename)
    savemat(save_path, {last_key: data_pca})
