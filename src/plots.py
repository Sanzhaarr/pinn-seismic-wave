import matplotlib.pyplot as plt
import numpy as np
import os


def plot_velocity_model(c, path="results/figures/velocity_model.png"):
    plt.figure(figsize=(6, 5))
    plt.imshow(c.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
    plt.colorbar(label="Velocity c(x,z)")
    plt.title("Heterogeneous Velocity Model")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_wave_snapshot(u, time_index, title, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(u[time_index].T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="seismic")
    plt.colorbar(label="Wave amplitude")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_comparison(fdm, pinn, time_index, path):
    error = np.abs(fdm[time_index] - pinn)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(fdm[time_index].T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="seismic")
    plt.title("FDM Reference")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(pinn.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="seismic")
    plt.title("PINN Prediction")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(error.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
    plt.title("Absolute Error")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_loss_curve(loss_history, path="results/figures/loss_curve.png"):
    plt.figure(figsize=(7, 5))
    plt.semilogy(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()