
import os

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _symmetric_limits(*arrays, percentile=99.0):
    values = []
    for array in arrays:
        array = np.asarray(array)
        if array.size > 0:
            values.append(np.abs(array).reshape(-1))

    if not values:
        return -1.0, 1.0

    joined = np.concatenate(values)
    limit = np.percentile(joined, percentile)

    if not np.isfinite(limit) or limit <= 0:
        limit = np.max(joined) if joined.size else 1.0

    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0

    return -float(limit), float(limit)


def plot_velocity_model(c, path="results/figures/velocity_model.png"):
    _ensure_parent_dir(path)

    plt.figure(figsize=(6.5, 5.2))
    image = plt.imshow(
        c.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
    )
    plt.colorbar(image, label="Velocity c(x,z)")
    plt.title("Heterogeneous Velocity Model")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_wave_snapshot(u, time_index, title, path):
    _ensure_parent_dir(path)

    snapshot = u[time_index]
    vmin, vmax = _symmetric_limits(snapshot)

    plt.figure(figsize=(6.5, 5.2))
    image = plt.imshow(
        snapshot.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(image, label="Amplitude")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison(fdm, pinn, time_index, path):
    _ensure_parent_dir(path)

    reference = fdm[time_index]
    prediction = np.asarray(pinn)
    error = np.abs(reference - prediction)

    vmin, vmax = _symmetric_limits(reference, prediction)
    error_max = np.percentile(error.reshape(-1), 99.0)
    if not np.isfinite(error_max) or error_max <= 0:
        error_max = np.max(error) if error.size else 1.0
    if not np.isfinite(error_max) or error_max <= 0:
        error_max = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    image0 = axes[0].imshow(
        reference.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("FDM Reference")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    fig.colorbar(image0, ax=axes[0], fraction=0.046, pad=0.04)

    image1 = axes[1].imshow(
        prediction.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("PINN Prediction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    fig.colorbar(image1, ax=axes[1], fraction=0.046, pad=0.04)

    image2 = axes[2].imshow(
        error.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        vmin=0.0,
        vmax=error_max,
    )
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("z")
    fig.colorbar(image2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(loss_history, path="results/figures/loss_curve.png"):
    _ensure_parent_dir(path)

    plt.figure(figsize=(7.2, 5.2))

    if isinstance(loss_history, dict):
        labels = {
            "total": "Total weighted loss",
            "pde": "PDE residual loss",
            "data": "Data loss",
            "ic": "Initial-condition loss",
            "bc": "Boundary-condition loss",
        }

        for key, label in labels.items():
            values = loss_history.get(key)
            if values:
                plt.semilogy(values, label=label)
    else:
        plt.semilogy(loss_history, label="Total loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss Components")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
