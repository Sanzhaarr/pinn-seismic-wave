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
from src.plots import (
    plot_baseline_comparison,
    plot_comparison,
    plot_loss_curve,
    plot_source_wavelet,
    plot_velocity_model,
    plot_wave_snapshot,
)
from src.pinn_model import SeismicPINN


def print_config(mode: str):
    print("\n===== PROJECT CONFIG CHECK =====")
    print(f"Mode: {mode}")
    print(f"Running config from: {cfg.__file__}")
    print(f"Config version: {getattr(cfg, 'CONFIG_VERSION', 'NO_CONFIG_VERSION_FOUND')}")
    print(f"Grid: NX={cfg.NX}, NZ={cfg.NZ}, NT={cfg.NT}")
    print(f"Training: EPOCHS={cfg.EPOCHS}, N_DATA={cfg.N_DATA}, N_COLLOCATION={cfg.N_COLLOCATION}")
    print(
        f"Loss weights: PDE={cfg.LAMBDA_PDE}, DATA={cfg.LAMBDA_DATA}, "
        f"IC={cfg.LAMBDA_IC}, BC={cfg.LAMBDA_BC}, AMP={getattr(cfg, 'LAMBDA_AMPLITUDE', 0.0)}"
    )
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
        {"parameter": "LAMBDA_AMPLITUDE", "value": getattr(cfg, "LAMBDA_AMPLITUDE", 0.0)},
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






def summarize_metric_table(df, model_name, role, training_time_seconds=np.nan):
    """Create one compact comparison row from a per-snapshot metric table."""
    return {
        "model": model_name,
        "role": role,
        "Mean_MSE": df["MSE"].mean(),
        "Mean_MAE": df["MAE"].mean(),
        "Mean_Relative_L2_Error": df["Relative_L2_Error"].mean(),
        "Mean_NRMSE": df["NRMSE"].mean(),
        "Mean_NMAE": df["NMAE"].mean() if "NMAE" in df.columns else np.nan,
        "Mean_Relative_Max_Error": df["Relative_Max_Error"].mean() if "Relative_Max_Error" in df.columns else np.nan,
        "Mean_PSNR_dB": df["PSNR_dB"].mean(),
        "Mean_Correlation": df["Correlation"].mean(),
        "Mean_Energy_Ratio": df["Energy_Ratio"].mean() if "Energy_Ratio" in df.columns else np.nan,
        "Mean_Bias": df["Mean_Bias"].mean() if "Mean_Bias" in df.columns else np.nan,
        "Training_time_seconds": training_time_seconds,
    }


def plot_metric_bar(summary_df, metric, path, title=None):
    """Save a bar chart comparing models using one metric."""
    import matplotlib.pyplot as plt

    if metric not in summary_df.columns:
        return

    plot_df = summary_df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(8, 4.8))
    plt.bar(plot_df["model"], plot_df[metric])
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title(title or f"Model Comparison: {metric}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_model_comparison_figures(comparison_df):
    """Save the most important dissertation comparison charts."""
    plot_metric_bar(
        comparison_df,
        "Mean_Relative_L2_Error",
        "results/figures/comparison_relative_l2_bar.png",
        "Relative L2 Error Comparison",
    )
    plot_metric_bar(
        comparison_df,
        "Mean_Correlation",
        "results/figures/comparison_correlation_bar.png",
        "Correlation Comparison",
    )
    plot_metric_bar(
        comparison_df,
        "Mean_Energy_Ratio",
        "results/figures/comparison_energy_ratio_bar.png",
        "Energy Ratio Comparison",
    )


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

def run_synthetic_mode(run_ablation=False):
    print("Step 1: Running heterogeneous finite difference simulation...")
    start_fdm = time.time()
    result = solve_wave_equation_fdm(model_type="heterogeneous", return_metadata=True)
    fdm_time = time.time() - start_fdm
    x, z, t, c, u_fdm, fdm_metadata = result

    print(f"FDM simulation completed in {fdm_time:.2f} seconds")
    print(f"FDM amplitude scale: {fdm_metadata['max_abs_amplitude']:.6e}")
    print(f"FDM CFL number: {fdm_metadata['cfl']:.6f}")
    print(f"Reference wavefield max amplitude: {max_abs_value(u_fdm):.6e}")

    print("Step 2: Running traditional finite difference baselines...")
    start_baseline = time.time()
    baseline_result = solve_wave_equation_fdm(model_type="homogeneous", return_metadata=True)
    baseline_time = time.time() - start_baseline
    _, _, _, c_homogeneous, u_homogeneous, baseline_metadata = baseline_result

    print(f"Homogeneous baseline completed in {baseline_time:.2f} seconds")
    print(f"Homogeneous baseline max amplitude: {baseline_metadata['max_abs_amplitude']:.6e}")

    start_layered = time.time()
    layered_result = solve_wave_equation_fdm(model_type="layered", return_metadata=True)
    layered_time = time.time() - start_layered
    _, _, _, c_layered, u_layered, layered_metadata = layered_result

    print(f"Layered baseline completed in {layered_time:.2f} seconds")
    print(f"Layered baseline max amplitude: {layered_metadata['max_abs_amplitude']:.6e}")

    pd.DataFrame([fdm_metadata, baseline_metadata, layered_metadata]).to_csv(
        "results/data/fdm_metadata.csv",
        index=False,
    )

    plot_velocity_model(c, path="results/figures/velocity_model.png")
    plot_velocity_model(
        c_homogeneous,
        path="results/figures/homogeneous_velocity_model.png",
    )
    plot_velocity_model(
        c_layered,
        path="results/figures/layered_velocity_model.png",
        title="Layered Baseline Velocity Model",
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
        plot_baseline_comparison(
            reference=u_fdm,
            baseline=u_layered,
            time_index=idx,
            path=f"results/figures/layered_baseline_comparison_t_{idx}.png",
        )

    neural_model_name = "data_only_nn" if cfg.LAMBDA_PDE == 0 else "weak_pde_pinn"
    neural_display_name = "Data-only NN" if cfg.LAMBDA_PDE == 0 else "Weak-PDE PINN"

    print(f"Step 3: Training synthetic neural model ({neural_display_name})...")
    start_pinn = time.time()
    model, loss_history = train_pinn(
        x,
        z,
        t,
        u_fdm,
        experiment_name=neural_model_name,
        lambda_pde=cfg.LAMBDA_PDE,
    )
    pinn_training_time = time.time() - start_pinn
    print(f"Neural model training completed in {pinn_training_time:.2f} seconds")

    plot_loss_curve(loss_history, path="results/figures/loss_curve.png")
    save_loss_history(loss_history, "results/data/loss_history.csv")

    residual_summary = pd.DataFrame([
        {
            "mode": neural_model_name,
            **evaluate_pde_residual(model),
        }
    ])
    residual_summary.to_csv("results/data/pde_residual_summary.csv", index=False)

    metrics_rows = []
    homogeneous_baseline_rows = []
    layered_baseline_rows = []

    for idx in snapshot_indices:
        pinn_snapshot = predict_snapshot(model, time_value=t[idx])
        snapshot_metrics = compute_all_metrics(pinn_snapshot, u_fdm[idx])
        homogeneous_baseline_metrics = compute_all_metrics(u_homogeneous[idx], u_fdm[idx])
        layered_baseline_metrics = compute_all_metrics(u_layered[idx], u_fdm[idx])

        metrics_rows.append(
            {
                "time_index": idx,
                "time": float(t[idx]),
                **snapshot_metrics,
            }
        )

        homogeneous_baseline_rows.append(
            {
                "time_index": idx,
                "time": float(t[idx]),
                **homogeneous_baseline_metrics,
            }
        )

        layered_baseline_rows.append(
            {
                "time_index": idx,
                "time": float(t[idx]),
                **layered_baseline_metrics,
            }
        )

        plot_comparison(
            fdm=u_fdm,
            pinn=pinn_snapshot,
            time_index=idx,
            path=f"results/figures/comparison_t_{idx}.png",
        )

    df = pd.DataFrame(metrics_rows)
    homogeneous_baseline_df = pd.DataFrame(homogeneous_baseline_rows)
    layered_baseline_df = pd.DataFrame(layered_baseline_rows)

    df.to_csv("results/data/metrics.csv", index=False)
    homogeneous_baseline_df.to_csv("results/data/homogeneous_baseline_metrics.csv", index=False)
    layered_baseline_df.to_csv("results/data/layered_baseline_metrics.csv", index=False)

    # Backward-compatible filename used by earlier report versions.
    homogeneous_baseline_df.to_csv("results/data/baseline_metrics.csv", index=False)

    comparison_rows = [
        summarize_metric_table(
            homogeneous_baseline_df,
            model_name="Homogeneous FDM baseline",
            role="traditional_simplified_baseline",
            training_time_seconds=baseline_time,
        ),
        summarize_metric_table(
            layered_baseline_df,
            model_name="Layered FDM baseline",
            role="traditional_layered_baseline",
            training_time_seconds=layered_time,
        ),
        summarize_metric_table(
            df,
            model_name=neural_display_name,
            role="neural_model",
            training_time_seconds=pinn_training_time,
        ),
    ]

    ablation_summary = None
    if run_ablation:
        ablation_summary = run_ablation_study(x, z, t, u_fdm, snapshot_indices)
        if ablation_summary is not None and not ablation_summary.empty:
            comparison_rows.extend(ablation_summary.to_dict("records"))

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv("results/data/model_comparison_summary.csv", index=False)
    save_model_comparison_figures(comparison_df)

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
            "Homogeneous_Mean_Relative_L2_Error": homogeneous_baseline_df["Relative_L2_Error"].mean(),
            "Homogeneous_Mean_NRMSE": homogeneous_baseline_df["NRMSE"].mean(),
            "Homogeneous_Mean_Correlation": homogeneous_baseline_df["Correlation"].mean(),
            "Layered_Mean_Relative_L2_Error": layered_baseline_df["Relative_L2_Error"].mean(),
            "Layered_Mean_NRMSE": layered_baseline_df["NRMSE"].mean(),
            "Layered_Mean_Correlation": layered_baseline_df["Correlation"].mean(),
            "FDM_time_seconds": fdm_time,
            "Homogeneous_FDM_time_seconds": baseline_time,
            "Layered_FDM_time_seconds": layered_time,
            "Neural_model_name": neural_model_name,
            "Neural_display_name": neural_display_name,
            "PINN_training_time_seconds": pinn_training_time,
        }
    ])

    summary_df.to_csv("results/data/synthetic_summary.csv", index=False)

    print("\nFinal synthetic quantitative results:")
    print(df)

    print("\nHomogeneous baseline quantitative results:")
    print(homogeneous_baseline_df)

    print("\nLayered baseline quantitative results:")
    print(layered_baseline_df)

    print("\nSynthetic summary:")
    print(summary_df)

    print("\nPDE residual diagnostics:")
    print(residual_summary)

    print("\nModel comparison summary:")
    print(comparison_df)

    return df, summary_df


def run_ablation_study(x, z, t, u_fdm, snapshot_indices):
    """Run optional smaller ablation experiments for dissertation comparison tables."""
    print("\nRunning optional ablation study...")
    ablation_epochs = max(500, cfg.EPOCHS // 2)
    experiments = [
        {
            "experiment": "data_only_no_fourier",
            "model": "Data-only NN without Fourier",
            "role": "ablation_fourier_off",
            "use_fourier": False,
            "lambda_pde": 0.0,
        },
        {
            "experiment": "weak_pde_fourier",
            "model": "Weak-PDE PINN",
            "role": "ablation_weak_physics",
            "use_fourier": True,
            "lambda_pde": 1e-7,
        },
    ]

    rows = []
    for exp in experiments:
        print(f"\nAblation experiment: {exp['experiment']}")
        start = time.time()
        model, loss_history = train_pinn(
            x,
            z,
            t,
            u_fdm,
            experiment_name=exp["experiment"],
            use_fourier=exp["use_fourier"],
            lambda_pde=exp["lambda_pde"],
            epochs=ablation_epochs,
        )
        training_time = time.time() - start
        save_loss_history(loss_history, f"results/data/loss_history_{exp['experiment']}.csv")

        metric_rows = []
        for idx in snapshot_indices:
            pred = predict_snapshot(model, time_value=t[idx])
            metric_rows.append(compute_all_metrics(pred, u_fdm[idx]))

        metric_df = pd.DataFrame(metric_rows)
        row = summarize_metric_table(
            metric_df,
            model_name=exp["model"],
            role=exp["role"],
            training_time_seconds=training_time,
        )
        row["lambda_pde"] = exp["lambda_pde"]
        row["use_fourier"] = exp["use_fourier"]
        row.update(evaluate_pde_residual(model))
        rows.append(row)

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv("results/data/ablation_summary.csv", index=False)
    return ablation_df


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
        "- results/figures/layered_baseline_comparison_t_*.png",
        "- results/figures/layered_velocity_model.png",
        "- results/figures/ricker_wavelet.png",
        "- results/figures/comparison_t_*.png",
        "- results/figures/loss_curve.png",
        "- results/data/metrics.csv",
        "- results/data/synthetic_summary.csv",
        "- results/data/homogeneous_baseline_metrics.csv",
        "- results/data/layered_baseline_metrics.csv",
        "- results/data/pde_residual_summary.csv",
        "- results/data/fdm_metadata.csv",
        "- results/data/model_comparison_summary.csv",
        "- results/figures/comparison_relative_l2_bar.png",
        "- results/figures/comparison_correlation_bar.png",
        "- results/figures/comparison_energy_ratio_bar.png",
        "- results/data/ablation_summary.csv, if ablation mode was executed",
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
            "Homogeneous and layered FDM simulations are included as traditional baselines. This allows the defense to compare the neural result against simplified physical models and against the heterogeneous reference solution with anomalies.",
            "",
        ]
    except ValueError:
        pass
    with open("results/data/defense_report.md", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def run_all_mode(run_ablation=False):
    print("Running full defense pipeline: synthetic experiment + real-data reconstruction if available.")
    synthetic_metrics, synthetic_summary = run_synthetic_mode(run_ablation=run_ablation)

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
    parser.add_argument("--ablation", action="store_true", help="Run optional ablation experiments. Slower but stronger for dissertation tables.")
    args = parser.parse_args()

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)

    print_config(args.mode)
    save_config_snapshot(args.mode)

    if args.mode == "real":
        real_summary = run_real_mode()
        write_defense_report(real_summary=real_summary)
    elif args.mode == "synthetic":
        _, synthetic_summary = run_synthetic_mode(run_ablation=args.ablation)
        synthetic_summary.to_csv("results/data/summary.csv", index=False)
        write_defense_report(synthetic_summary=synthetic_summary)
    else:
        run_all_mode(run_ablation=args.ablation)

    print("\nProject completed.")
    print("Generated figures are saved in: results/figures")
    print("Results are saved in: results/data")


if __name__ == "__main__":
    main()