import numpy as np


def mse_error(pred, true):
    return np.mean((pred - true) ** 2)


def mae_error(pred, true):
    return np.mean(np.abs(pred - true))


def relative_l2_error(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)