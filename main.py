import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config as cfg
from src.fdm_solver import solve_wave_equation_fdm
from src.train_pinn import train_pinn, predict_snapshot
from src.metrics import compute_all_metrics, max_abs_value
from src.plots import plot_velocity_model, plot_wave_snapshot, plot_comparison, plot_loss_curve
from src.pinn_model import SeismicPINN


def print_config(mode: str):
    print("\n===== PROJECT CONFIG CHECK =====")
    print(f"Mode: {mode}")
    print(f"Running config from: {cfg.__file__}")
    print(f"Config version: {getattr(cfg, 'CONFIG_VERSION', 'NO_CONFIG_VERSION_FOUND')}")
    print(f"Grid: NX={cfg.NX}, NZ={cfg.NZ}, NT={cfg.NT}")
    print(f"Training: EPOCHS={cfg.EPOCHS}, N_DATA={cfg.N_DATA}, N_COLLOCATION={cfg.N_COLLOCATION}")
    print(f"Loss weights: PDE={cfg.LAMBDA_PDE}, DATA={cfg.LAMBDA_DATA}, IC={cfg.LAMBDA_IC}, BC={cfg.LAMBDA_BC}")
    print("================================\n")


def load_real_section():
    real_dir = Path("data/real")
    candidates = [
        real_dir / "real_section.npy",
        real_dir / "real_section.csv",
        real_dir / "real_section.sgy",
        real_dir / "real_section.segy",
    ]

    for path in candidates:
        if not path.exists():
            continue

        print(f"Found real seismic file: {path}")
        suffix = path.suffix.lower()

        if suffix == ".npy":
            section = np.load(path)
        elif suffix == ".csv":
            section = np.loadtxt(path, delimiter=",")
        elif suffix in [".sgy", ".segy"]:
            try:
                import segyio
            except ImportError as exc:
                raise ImportError("Install segyio first: pip install segyio") from exc
            with segyio.open(str(path), ignore_geometry=True) as f:
                section = segyio.tools.collect(f.trace[:])
        else:
            continue

        section = np.asarray(section, dtype=np.float32)
        section = np.squeeze(section)

        if section.ndim != 2:
            raise ValueError(f"Real seismic section must be 2D, got shape={section.shape}")

        print(f"Original loaded section shape: {section.shape}")

        section = np.nan_to_num(section)
        section = section - np.mean(section)

        sx, sz = section.shape
        nx, nz = cfg.NX, cfg.NZ
        start_x = max((sx - nx) // 2, 0)
        start_z = max((sz - nz) // 2, 0)
        cropped = section[start_x:start_x + nx, start_z:start_z + nz]

        out = np.zeros((nx, nz), dtype=np.float32)
        cx, cz = cropped.shape
        out[:cx, :cz] = cropped

        scale = float(np.max(np.abs(out)))
        if scale < 1e-12:
            raise ValueError("Loaded real seismic section has near-zero amplitude after crop.")

        out = out / scale

        print(f"Prepared real section shape: {out.shape}")
        print(f"Amplitude after normalization: min={out.min():.6f}, max={out.max():.6f}")

        return out, path, scale

    raise FileNotFoundError(
        "No real seismic file found. Put one of these files into data/real/:\n"
        "  real_section.npy\n"
        "  real_section.csv\n"
        "  real_section.sgy\n"
        "  real_section.segy"
    )


def train_real_section_pinn(section):
    model = SeismicPINN(
        use_fourier=cfg.USE_FOURIER_FEATURES,
        mapping_size=cfg.FOURIER_MAPPING_SIZE,
        fourier_scale=cfg.FOURIER_SCALE,
        hidden_dim=cfg.HIDDEN_DIM,
        depth=cfg.NETWORK_DEPTH,
        activation=cfg.ACTIVATION,
    ).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.5)

    nx, nz = section.shape
    x = np.linspace(0.0, 1.0, nx)
    z = np.linspace(0.0, 1.0, nz)
    X, Z = np.meshgrid(x, z, indexing="ij")
    T = np.ones_like(X) * 0.5

    coords = np.stack([X.reshape(-1), Z.reshape(-1), T.reshape(-1)], axis=1)
    values = section.reshape(-1, 1)

    coords_t = torch.tensor(coords, dtype=torch.float32).to(cfg.DEVICE)
    values_t = torch.tensor(values, dtype=torch.float32).to(cfg.DEVICE)

    n_total = coords_t.shape[0]
    batch_size = min(4096, n_total)

    loss_history = {"total": [], "pde": [], "data": []}

    for epoch in range(cfg.EPOCHS):
        idx = torch.randint(0, n_total, (batch_size,), device=cfg.DEVICE)
        batch_xzt = coords_t[idx].clone().detach().requires_grad_(True)
        batch_y = values_t[idx]

        optimizer.zero_grad()
        pred = model(batch_xzt)
        loss_data = torch.mean((pred - batch_y) ** 2)

        grads = torch.autograd.grad(
            pred,
            batch_xzt,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
        )[0]

        u_x = grads[:, 0:1]
        u_z = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x,
            batch_xzt,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
        )[0][:, 0:1]

        u_zz = torch.autograd.grad(
            u_z,
            batch_xzt,
            grad_outputs=torch.ones_like(u_z),
            create_graph=True,
        )[0][:, 1:2]

        loss_pde = torch.mean((u_xx + u_zz) ** 2)
        loss = cfg.LAMBDA_DATA * loss_data + cfg.LAMBDA_PDE * loss_pde

        loss.backward()
        if cfg.GRADIENT_CLIP_NORM is not None and cfg.GRADIENT_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRADIENT_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        loss_history["total"].append(float(loss.item()))
        loss_history["pde"].append(float(loss_pde.item()))
        loss_history["data"].append(float(loss_data.item()))

        if epoch % 250 == 0:
            print(
                f"Epoch {epoch} | "
                f"Total: {loss.item():.6e} | "
                f"Smoothness/PDE: {loss_pde.item():.6e} | "
                f"Data: {loss_data.item():.6e}"
            )

    model.eval()
    with torch.no_grad():
        pred = model(coords_t).cpu().numpy().reshape(nx, nz)

    torch.save(model.state_dict(), "results/data/real_section_pinn_model.pt")
    return pred, loss_history


def run_real_mode():
    print("Step 1: Loading real seismic section...")
    section, source_path, scale = load_real_section()

    print(f"Loaded real seismic section from: {source_path}")
    print(f"Original real-data amplitude scale: {scale:.6e}")
    print(f"Normalized section max amplitude: {max_abs_value(section):.6e}")

    section_volume = section[None, :, :]
    plot_wave_snapshot(
        section_volume,
        0,
        title="Normalized Real Seismic Section",
        path="results/figures/real_section_reference.png",
    )

    print("Step 2: Training real-data PINN...")
    start = time.time()
    pred, loss_history = train_real_section_pinn(section)
    training_time = time.time() - start
    print(f"Real-data PINN training completed in {training_time:.2f} seconds")

    print("Step 3: Saving real-data comparison and metrics...")
    plot_comparison(
        fdm=section_volume,
        pinn=pred,
        time_index=0,
        path="results/figures/real_section_comparison.png",
    )
    plot_loss_curve(loss_history, path="results/figures/real_loss_curve.png")

    real_metrics = compute_all_metrics(pred, section)
    df = pd.DataFrame([
        {
            "mode": "real",
            "source_file": str(source_path),
            **real_metrics,
            "training_time_seconds": training_time,
        }
    ])

    df.to_csv("results/data/real_metrics.csv", index=False)
    df.to_csv("results/data/summary.csv", index=False)

    print("\nFinal real-data quantitative results:")
    print(df)


def run_synthetic_mode():
    print("Step 1: Running finite difference simulation...")
    start_fdm = time.time()
    result = solve_wave_equation_fdm()
    fdm_time = time.time() - start_fdm

    if len(result) == 6:
        x, z, t, c, u_fdm, fdm_scale = result
    else:
        x, z, t, c, u_fdm = result
        fdm_scale = max_abs_value(u_fdm)

    print(f"FDM simulation completed in {fdm_time:.2f} seconds")
    print(f"FDM amplitude scale: {fdm_scale:.6e}")
    print(f"Reference wavefield max amplitude: {max_abs_value(u_fdm):.6e}")

    plot_velocity_model(c)
    snapshot_indices = sorted(set([
        max(0, min(len(t) - 1, int(0.25 * len(t)))),
        max(0, min(len(t) - 1, int(0.45 * len(t)))),
        max(0, min(len(t) - 1, int(0.65 * len(t)))),
        max(0, min(len(t) - 1, int(0.85 * len(t)))),
    ]))

    for idx in snapshot_indices:
        plot_wave_snapshot(
            u_fdm,
            idx,
            title=f"FDM Wavefield Snapshot at t={t[idx]:.3f}",
            path=f"results/figures/fdm_snapshot_{idx}.png",
        )

    print("Step 3: Training synthetic PINN...")
    start_pinn = time.time()
    model, loss_history = train_pinn(x, z, t, u_fdm)
    pinn_training_time = time.time() - start_pinn
    print(f"PINN training completed in {pinn_training_time:.2f} seconds")

    metrics_rows = []
    for idx in snapshot_indices:
        pinn_snapshot = predict_snapshot(model, time_value=t[idx])
        snapshot_metrics = compute_all_metrics(pinn_snapshot, u_fdm[idx])
        metrics_rows.append(
            {
                "time_index": idx,
                "time": float(t[idx]),
                **snapshot_metrics,
            }
        )
        plot_comparison(
            fdm=u_fdm,
            pinn=pinn_snapshot,
            time_index=idx,
            path=f"results/figures/comparison_t_{idx}.png",
        )

    plot_loss_curve(loss_history)
    df = pd.DataFrame(metrics_rows)
    df.to_csv("results/data/metrics.csv", index=False)

    summary_df = pd.DataFrame([
        {
            "mode": "synthetic",
            "num_snapshots": len(snapshot_indices),
            "Mean_MSE": df["MSE"].mean(),
            "Mean_MAE": df["MAE"].mean(),
            "Mean_Relative_L2_Error": df["Relative_L2_Error"].mean(),
            "Mean_NRMSE": df["NRMSE"].mean(),
            "Mean_PSNR_dB": df["PSNR_dB"].mean(),
            "Mean_Correlation": df["Correlation"].mean(),
            "FDM_time_seconds": fdm_time,
            "PINN_training_time_seconds": pinn_training_time,
        }
    ])
    summary_df.to_csv("results/data/summary.csv", index=False)

    print("\nFinal synthetic quantitative results:")
    print(df)
    print("\nSynthetic summary:")
    print(summary_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    args = parser.parse_args()

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)

    print_config(args.mode)

    if args.mode == "real":
        run_real_mode()
    else:
        run_synthetic_mode()

    print("\nProject completed.")
    print("Generated figures are saved in: results/figures")
    print("Results are saved in: results/data")


if __name__ == "__main__":
    main()