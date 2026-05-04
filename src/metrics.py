import numpy as np


EPS = 1e-12


def _as_float_arrays(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    if pred.shape != true.shape:
        raise ValueError(f"Prediction and reference shapes must match. Got {pred.shape} and {true.shape}.")

    return pred, true


def mse_error(pred, true):
    pred, true = _as_float_arrays(pred, true)
    return float(np.mean((pred - true) ** 2))


def mae_error(pred, true):
    pred, true = _as_float_arrays(pred, true)
    return float(np.mean(np.abs(pred - true)))


def relative_l2_error(pred, true):
    pred, true = _as_float_arrays(pred, true)
    true_norm = np.linalg.norm(true)
    if true_norm < EPS:
        return np.nan
    return float(np.linalg.norm(pred - true) / true_norm)


def max_abs_value(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return np.nan
    return float(np.max(np.abs(arr)))


def normalized_rmse(pred, true):
    pred, true = _as_float_arrays(pred, true)
    scale = np.max(np.abs(true))
    if scale < EPS:
        return np.nan
    return float(np.sqrt(np.mean((pred - true) ** 2)) / scale)


def peak_signal_to_noise_ratio(pred, true):
    pred, true = _as_float_arrays(pred, true)
    mse = np.mean((pred - true) ** 2)
    peak = np.max(np.abs(true))

    if mse < EPS:
        return np.inf
    if peak < EPS:
        return np.nan

    return float(20.0 * np.log10(peak / np.sqrt(mse)))


def correlation_coefficient(pred, true):
    pred, true = _as_float_arrays(pred, true)

    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    pred_std = np.std(pred_flat)
    true_std = np.std(true_flat)

    if pred_std < EPS or true_std < EPS:
        return np.nan

    return float(np.corrcoef(pred_flat, true_flat)[0, 1])


def compute_all_metrics(pred, true):
    return {
        "MSE": mse_error(pred, true),
        "MAE": mae_error(pred, true),
        "Relative_L2_Error": relative_l2_error(pred, true),
        "NRMSE": normalized_rmse(pred, true),
        "PSNR_dB": peak_signal_to_noise_ratio(pred, true),
        "Correlation": correlation_coefficient(pred, true),
        "Reference_Max_Abs": max_abs_value(true),
        "Prediction_Max_Abs": max_abs_value(pred),
        "Error_Max_Abs": max_abs_value(np.asarray(pred) - np.asarray(true)),
    }