import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config as cfg
from src.fdm_solver import solve_wave_equation_fdm
from src.train_pinn import evaluate_pde_residual, predict_snapshot, train_pinn
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


def save_config_snapshot(mode: str):
    config_rows = [
        {"parameter": "mode", "value": mode},
        {"parameter": "config_version", "value": getattr(cfg, "CONFIG_VERSION", "unknown")},
        {"parameter": "device", "value": str(cfg.DEVICE)},
        {"parameter": "NX", "value": cfg.NX},
        {"parameter": "NZ", "value": cfg.NZ},
        {"parameter": "NT", "value": cfg.NT},
        {"parameter": "T_MAX", "value": cfg.T_MAX},
        {"parameter": "SOURCE_FREQUENCY", "value": cfg.SOURCE_FREQUENCY},
        {"parameter": "SOURCE_AMPLITUDE", "value": cfg.SOURCE_AMPLITUDE},
        {"parameter": "USE_FOURIER_FEATURES", "value": cfg.USE_FOURIER_FEATURES},
        {"parameter": "FOURIER_MAPPING_SIZE", "value": cfg.FOURIER_MAPPING_SIZE},
        {"parameter": "FOURIER_SCALE", "value": cfg.FOURIER_SCALE},
        {"parameter": "HIDDEN_DIM", "value": cfg.HIDDEN_DIM},
        {"parameter": "NETWORK_DEPTH", "value": cfg.NETWORK_DEPTH},
        {"parameter": "ACTIVATION", "value": cfg.ACTIVATION},
        {"parameter": "EPOCHS", "value": cfg.EPOCHS},
        {"parameter": "N_DATA", "value": cfg.N_DATA},
        {"parameter": "N_COLLOCATION", "value": cfg.N_COLLOCATION},
        {"parameter": "N_INITIAL", "value": cfg.N_INITIAL},
        {"parameter": "N_BOUNDARY", "value": cfg.N_BOUNDARY},
        {"parameter": "LAMBDA_PDE", "value": cfg.LAMBDA_PDE},
        {"parameter": "LAMBDA_DATA", "value": cfg.LAMBDA_DATA},
        {"parameter": "LAMBDA_IC", "value": cfg.LAMBDA_IC},
        {"parameter": "LAMBDA_BC", "value": cfg.LAMBDA_BC},
    ]
    pd.DataFrame(config_rows).to_csv(f"results/data/config_snapshot_{mode}.csv", index=False)


# ==================== Helper functions for baseline and results ====================

def get_snapshot_indices(t):
    """Choose six representative time indices for figures and metrics."""
    percentages = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    return sorted(set([
        max(0, min(len(t) - 1, int(p * len(t)))) for p in percentages
    ]))


def save_loss_history(loss_history, path):
    """Persist loss history as CSV for dissertation tables and debugging."""
    pd.DataFrame(loss_history).to_csv(path, index=False)


def plot_baseline_comparison(reference, baseline, time_index, path):
    """Plot heterogeneous FDM reference against homogeneous FDM baseline."""
    import matplotlib.pyplot as plt

    ref = reference[time_index]
    base = baseline[time_index]
    error = np.abs(ref - base)

    amplitude_limit = np.percentile(
        np.abs(np.concatenate([ref.reshape(-1), base.reshape(-1)])),
        99.0,
    )
    if not np.isfinite(amplitude_limit) or amplitude_limit <= 0:
        amplitude_limit = max(max_abs_value(ref), max_abs_value(base), 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    im0 = axes[0].imshow(
        ref.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="seismic",
        vmin=-amplitude_limit,
        vmax=amplitude_limit,
    )
    axes[0].set_title("Heterogeneous FDM Reference")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        base.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="seismic",
        vmin=-amplitude_limit,
        vmax=amplitude_limit,
    )
    axes[1].set_title("Homogeneous FDM Baseline")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        error.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
    )
    axes[2].set_title("Baseline Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("z")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_source_wavelet(t, path="results/figures/ricker_wavelet.png"):
    """Plot the configured Ricker wavelet source."""
    import matplotlib.pyplot as plt
    from src.fdm_solver import ricker_wavelet

    plt.figure(figsize=(7, 4.5))
    plt.plot(t, ricker_wavelet(t))
    plt.xlabel("Time")
    plt.ylabel("Source amplitude")
    plt.title("Ricker Wavelet Source")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


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
    save_loss_history(loss_history, "results/data/real_loss_history.csv")

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
    df.to_csv("results/data/real_summary.csv", index=False)

    print("\nFinal real-data quantitative results:")
    print(df)

    return df

def run_synthetic_mode():
    print("Step 1: Running heterogeneous finite difference simulation...")
    start_fdm = time.time()
    result = solve_wave_equation_fdm(model_type="heterogeneous", return_metadata=True)
    fdm_time = time.time() - start_fdm
    x, z, t, c, u_fdm, fdm_metadata = result

    print(f"FDM simulation completed in {fdm_time:.2f} seconds")
    print(f"FDM amplitude scale: {fdm_metadata['max_abs_amplitude']:.6e}")
    print(f"FDM CFL number: {fdm_metadata['cfl']:.6f}")
    print(f"Reference wavefield max amplitude: {max_abs_value(u_fdm):.6e}")

    print("Step 2: Running homogeneous finite difference baseline...")
    start_baseline = time.time()
    baseline_result = solve_wave_equation_fdm(model_type="homogeneous", return_metadata=True)
    baseline_time = time.time() - start_baseline
    _, _, _, c_homogeneous, u_homogeneous, baseline_metadata = baseline_result

    print(f"Homogeneous baseline completed in {baseline_time:.2f} seconds")
    print(f"Homogeneous baseline max amplitude: {baseline_metadata['max_abs_amplitude']:.6e}")

    pd.DataFrame([fdm_metadata, baseline_metadata]).to_csv(
        "results/data/fdm_metadata.csv",
        index=False,
    )

    plot_velocity_model(c, path="results/figures/velocity_model.png")
    plot_velocity_model(
        c_homogeneous,
        path="results/figures/homogeneous_velocity_model.png",
    )
    plot_source_wavelet(t)

    snapshot_indices = get_snapshot_indices(t)

    for idx in snapshot_indices:
        plot_wave_snapshot(
            u_fdm,
            idx,
            title=f"FDM Wavefield Snapshot at t={t[idx]:.3f}",
            path=f"results/figures/fdm_snapshot_{idx}.png",
        )
        plot_baseline_comparison(
            reference=u_fdm,
            baseline=u_homogeneous,
            time_index=idx,
            path=f"results/figures/baseline_comparison_t_{idx}.png",
        )

    print("Step 3: Training synthetic PINN...")
    start_pinn = time.time()
    model, loss_history = train_pinn(x, z, t, u_fdm, experiment_name="full")
    pinn_training_time = time.time() - start_pinn
    print(f"PINN training completed in {pinn_training_time:.2f} seconds")

    plot_loss_curve(loss_history, path="results/figures/loss_curve.png")
    save_loss_history(loss_history, "results/data/loss_history.csv")

    residual_summary = pd.DataFrame([
        {
            "mode": "synthetic_full",
            **evaluate_pde_residual(model),
        }
    ])
    residual_summary.to_csv("results/data/pde_residual_summary.csv", index=False)

    metrics_rows = []
    baseline_rows = []

    for idx in snapshot_indices:
        pinn_snapshot = predict_snapshot(model, time_value=t[idx])
        snapshot_metrics = compute_all_metrics(pinn_snapshot, u_fdm[idx])
        baseline_metrics = compute_all_metrics(u_homogeneous[idx], u_fdm[idx])

        metrics_rows.append(
            {
                "time_index": idx,
                "time": float(t[idx]),
                **snapshot_metrics,
            }
        )

        baseline_rows.append(
            {
                "time_index": idx,
                "time": float(t[idx]),
                **baseline_metrics,
            }
        )

        plot_comparison(
            fdm=u_fdm,
            pinn=pinn_snapshot,
            time_index=idx,
            path=f"results/figures/comparison_t_{idx}.png",
        )

    df = pd.DataFrame(metrics_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    df.to_csv("results/data/metrics.csv", index=False)
    baseline_df.to_csv("results/data/baseline_metrics.csv", index=False)

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
            "Baseline_Mean_Relative_L2_Error": baseline_df["Relative_L2_Error"].mean(),
            "Baseline_Mean_NRMSE": baseline_df["NRMSE"].mean(),
            "Baseline_Mean_Correlation": baseline_df["Correlation"].mean(),
            "FDM_time_seconds": fdm_time,
            "Homogeneous_FDM_time_seconds": baseline_time,
            "PINN_training_time_seconds": pinn_training_time,
        }
    ])

    summary_df.to_csv("results/data/synthetic_summary.csv", index=False)

    print("\nFinal synthetic quantitative results:")
    print(df)

    print("\nHomogeneous baseline quantitative results:")
    print(baseline_df)

    print("\nSynthetic summary:")
    print(summary_df)

    print("\nPDE residual diagnostics:")
    print(residual_summary)

    return df, summary_df


def write_defense_report(synthetic_summary=None, real_summary=None):
    lines = [
        "# Defense Results Summary",
        "",
        "Project topic:",
        "Физически-информированные нейронные сети для моделирования распространения сейсмических волн в неоднородной среде",
        "",
        "## Main proof used in the defense",
        "",
        "The main proof is the controlled synthetic experiment. In this experiment, the heterogeneous velocity model, seismic source, initial conditions, boundary treatment, and finite-difference reference solution are known. This makes the comparison between FDM and PINN scientifically defensible.",
        "",
        "## Secondary real-data demonstration",
        "",
        "The real seismic section is included as an applied demonstration. It shows that the neural representation can reconstruct a real seismic section, but it should be described as real-data reconstruction with physics-inspired smoothness regularization unless the exact field velocity model, acquisition geometry, and source wavelet are available.",
        "",
        "## Recommended defense wording",
        "",
        "The proposed PINN framework was evaluated in a controlled heterogeneous-medium experiment by comparing the neural approximation with a finite-difference reference solution of the acoustic wave equation. The real seismic section was then used as an additional reconstruction experiment to demonstrate practical applicability on field-like data.",
        "",
    ]

    if synthetic_summary is not None:
        lines.extend([
            "## Synthetic experiment summary",
            "",
            synthetic_summary.to_string(index=False),
            "",
        ])

    if real_summary is not None:
        lines.extend([
            "## Real-data reconstruction summary",
            "",
            real_summary.to_string(index=False),
            "",
        ])

    lines.extend([
        "## Files to show during defense",
        "",
        "- results/figures/velocity_model.png",
        "- results/figures/fdm_snapshot_*.png",
        "- results/figures/baseline_comparison_t_*.png",
        "- results/figures/ricker_wavelet.png",
        "- results/figures/comparison_t_*.png",
        "- results/figures/loss_curve.png",
        "- results/data/metrics.csv",
        "- results/data/synthetic_summary.csv",
        "- results/data/baseline_metrics.csv",
        "- results/data/pde_residual_summary.csv",
        "- results/data/fdm_metadata.csv",
        "- results/data/real_metrics.csv, if real mode was executed",
        "",
        "## Important limitation",
        "",
        "The real-data experiment alone is not sufficient as proof of physical wave propagation because the true velocity model, source function, and acquisition geometry are not fully constrained. The synthetic FDM-vs-PINN experiment should therefore remain the main scientific validation.",
    ])

    # Insert extra lines about baseline validation after the main proof paragraph
    main_proof_str = (
        "The proposed PINN framework was evaluated in a controlled heterogeneous-medium experiment by comparing the neural approximation with a finite-difference reference solution of the acoustic wave equation. The real seismic section was then used as an additional reconstruction experiment to demonstrate practical applicability on field-like data."
    )
    try:
        idx = lines.index(main_proof_str)
        lines[idx+1:idx+1] = [
            "## Added baseline validation",
            "",
            "A homogeneous-medium FDM simulation is included as a baseline. This allows the defense to compare the proposed PINN result not only against the heterogeneous reference solution, but also against a simplified physical model that ignores subsurface heterogeneity.",
            "",
        ]
    except ValueError:
        pass
    with open("results/data/defense_report.md", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def run_all_mode():
    print("Running full defense pipeline: synthetic experiment + real-data reconstruction if available.")
    synthetic_metrics, synthetic_summary = run_synthetic_mode()

    real_summary = None
    try:
        real_summary = run_real_mode()
    except Exception as exc:
        print("\nReal-data mode was skipped because it failed:")
        print(str(exc))
        print("Synthetic results are still valid as the main defense proof.")

    summaries = [synthetic_summary]
    if real_summary is not None:
        summaries.append(real_summary)
    pd.concat(summaries, ignore_index=True, sort=False).to_csv("results/data/summary.csv", index=False)

    write_defense_report(synthetic_summary=synthetic_summary, real_summary=real_summary)
    return synthetic_metrics, synthetic_summary, real_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real", "all"], default="all")
    args = parser.parse_args()

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)

    print_config(args.mode)
    save_config_snapshot(args.mode)

    if args.mode == "real":
        real_summary = run_real_mode()
        write_defense_report(real_summary=real_summary)
    elif args.mode == "synthetic":
        _, synthetic_summary = run_synthetic_mode()
        synthetic_summary.to_csv("results/data/summary.csv", index=False)
        write_defense_report(synthetic_summary=synthetic_summary)
    else:
        run_all_mode()

    print("\nProject completed.")
    print("Generated figures are saved in: results/figures")
    print("Results are saved in: results/data")


if __name__ == "__main__":
    main()