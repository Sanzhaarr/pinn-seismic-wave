import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config as cfg
from src.advanced_cases import apparent_velocity_diagnostic, write_advanced_scope_report
from src.fdm_solver import normalize_sources, solve_wave_equation_fdm
from src.train_pinn import evaluate_pde_residual, predict_snapshot, train_pinn
from src.metrics import compute_all_metrics, max_abs_value
from src.thesis_assets import write_thesis_assets
from src.plots import (
    plot_baseline_comparison,
    plot_comparison,
    plot_loss_curve,
    plot_source_wavelet,
    plot_velocity_model,
    plot_wave_snapshot,
)
from src.pinn_model import SeismicPINN


def print_config(mode: str, scenario="heterogeneous", source_case="single", run_multi_seed=False):
    print("\n===== PROJECT CONFIG CHECK =====")
    print(f"Mode: {mode}")
    print(f"Running config from: {cfg.__file__}")
    print(f"Config version: {getattr(cfg, 'CONFIG_VERSION', 'NO_CONFIG_VERSION_FOUND')}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Synthetic scenario request: {scenario}")
    print(f"Source case: {source_case}")
    print(f"Grid: NX={cfg.NX}, NZ={cfg.NZ}, NT={cfg.NT}")
    print(f"Coarse FDM baseline grid: NX={cfg.COARSE_NX}, NZ={cfg.COARSE_NZ}, NT={cfg.COARSE_NT}")
    print(f"Training: EPOCHS={cfg.EPOCHS}, N_DATA={cfg.N_DATA}, N_COLLOCATION={cfg.N_COLLOCATION}")
    print(
        f"Loss weights: PDE={cfg.LAMBDA_PDE}, DATA={cfg.LAMBDA_DATA}, "
        f"IC={cfg.LAMBDA_IC}, BC={cfg.LAMBDA_BC}, AMP={getattr(cfg, 'LAMBDA_AMPLITUDE', 0.0)}"
    )
    if run_multi_seed:
        print(f"Multi-seed repeats: seeds={cfg.MULTI_SEED_VALUES}, epochs={cfg.MULTI_SEED_EPOCHS}")
    print("================================\n")


def save_config_snapshot(mode: str, scenario="heterogeneous", source_case="single", run_multi_seed=False):
    config_rows = [
        {"parameter": "mode", "value": mode},
        {"parameter": "scenario_request", "value": scenario},
        {"parameter": "source_case", "value": source_case},
        {"parameter": "multi_seed_requested", "value": run_multi_seed},
        {"parameter": "config_version", "value": getattr(cfg, "CONFIG_VERSION", "unknown")},
        {"parameter": "device", "value": str(cfg.DEVICE)},
        {"parameter": "NX", "value": cfg.NX},
        {"parameter": "NZ", "value": cfg.NZ},
        {"parameter": "NT", "value": cfg.NT},
        {"parameter": "COARSE_NX", "value": cfg.COARSE_NX},
        {"parameter": "COARSE_NZ", "value": cfg.COARSE_NZ},
        {"parameter": "COARSE_NT", "value": cfg.COARSE_NT},
        {"parameter": "SPARSE_BASELINE_TIME_STRIDE", "value": cfg.SPARSE_BASELINE_TIME_STRIDE},
        {"parameter": "SPARSE_BASELINE_SPACE_STRIDE", "value": cfg.SPARSE_BASELINE_SPACE_STRIDE},
        {"parameter": "SYNTHETIC_SCENARIOS", "value": ",".join(cfg.SYNTHETIC_SCENARIOS)},
        {"parameter": "MULTI_SOURCE_COUNT", "value": len(cfg.MULTI_SOURCE_DEFINITIONS)},
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
        {"parameter": "PDE_WARMUP_EPOCHS", "value": getattr(cfg, "PDE_WARMUP_EPOCHS", 0)},
        {"parameter": "REAL_SMOOTHNESS_WEIGHT", "value": getattr(cfg, "REAL_SMOOTHNESS_WEIGHT", 0.0)},
        {"parameter": "ABLATION_EPOCHS", "value": getattr(cfg, "ABLATION_EPOCHS", "auto")},
        {"parameter": "MULTI_SEED_VALUES", "value": ",".join(str(seed) for seed in cfg.MULTI_SEED_VALUES)},
        {"parameter": "MULTI_SEED_EPOCHS", "value": getattr(cfg, "MULTI_SEED_EPOCHS", "auto")},
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


def save_model_comparison_figures(comparison_df, scenario="heterogeneous"):
    """Save the most important dissertation comparison charts."""
    plot_metric_bar(
        comparison_df,
        "Mean_Relative_L2_Error",
        figure_path("comparison_relative_l2_bar.png", scenario=scenario),
        "Relative L2 Error Comparison",
    )
    plot_metric_bar(
        comparison_df,
        "Mean_Correlation",
        figure_path("comparison_correlation_bar.png", scenario=scenario),
        "Correlation Comparison",
    )
    plot_metric_bar(
        comparison_df,
        "Mean_Energy_Ratio",
        figure_path("comparison_energy_ratio_bar.png", scenario=scenario),
        "Energy Ratio Comparison",
    )


def scenario_suffix(scenario):
    return "" if scenario == "heterogeneous" else f"_{scenario}"


def source_case_suffix(source_case):
    return "" if source_case == "single" else f"_{source_case}"


def artifact_key(scenario, source_case="single"):
    return f"{scenario}{source_case_suffix(source_case)}"


def scenario_label(scenario):
    labels = {
        "heterogeneous": "Layered anomalies",
        "faulted": "Faulted channel/lens",
    }
    return labels.get(scenario, scenario.replace("_", " ").title())


def artifact_path(directory, filename, scenario="heterogeneous"):
    """Return a scenario-aware path while preserving legacy names for the default scenario."""
    suffix = scenario_suffix(scenario)
    if not suffix:
        return f"{directory}/{filename}"

    stem, ext = os.path.splitext(filename)
    return f"{directory}/{stem}{suffix}{ext}"


def data_path(filename, scenario="heterogeneous"):
    return artifact_path("results/data", filename, scenario=scenario)


def figure_path(filename, scenario="heterogeneous"):
    return artifact_path("results/figures", filename, scenario=scenario)


def resolve_scenarios(scenario_arg):
    if scenario_arg == "all":
        return list(cfg.SYNTHETIC_SCENARIOS)
    if scenario_arg not in cfg.SYNTHETIC_SCENARIOS:
        raise ValueError(
            f"Unknown scenario={scenario_arg!r}. Valid values: {cfg.SYNTHETIC_SCENARIOS + ['all']}"
        )
    return [scenario_arg]


def resolve_source_case(source_case):
    if source_case not in {"single", "multi_source"}:
        raise ValueError("source_case must be one of: 'single', 'multi_source'")
    return source_case


def get_sources_for_case(source_case):
    source_case = resolve_source_case(source_case)
    if source_case == "multi_source":
        return normalize_sources(cfg.MULTI_SOURCE_DEFINITIONS)
    return normalize_sources(None)


def resample_wavefield_to_grid(u, source_t, source_x, source_z, target_t, target_x, target_z):
    """Interpolate a wavefield volume from one regular FDM grid to another."""

    def interp_along_axis(values, source_grid, target_grid, axis):
        moved = np.moveaxis(values, axis, 0)
        original_shape = moved.shape
        flat = moved.reshape(original_shape[0], -1)
        out = np.empty((len(target_grid), flat.shape[1]), dtype=np.float32)

        for column in range(flat.shape[1]):
            out[:, column] = np.interp(
                target_grid,
                source_grid,
                flat[:, column],
                left=0.0,
                right=0.0,
            )

        out = out.reshape((len(target_grid),) + original_shape[1:])
        return np.moveaxis(out, 0, axis)

    resampled = interp_along_axis(u, source_t, target_t, axis=0)
    resampled = interp_along_axis(resampled, source_x, target_x, axis=1)
    resampled = interp_along_axis(resampled, source_z, target_z, axis=2)
    return resampled.astype(np.float32)


def sparse_indices(length, stride):
    """Return sparse monotonically increasing sample indices including both endpoints."""
    stride = max(1, int(stride))
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return np.array(indices, dtype=int)


def build_sparse_interpolation_baseline(u_fdm, x, z, t, time_stride, space_stride):
    """Build a traditional sparse-sample interpolation baseline."""
    t_idx = sparse_indices(len(t), time_stride)
    x_idx = sparse_indices(len(x), space_stride)
    z_idx = sparse_indices(len(z), space_stride)

    sparse_volume = u_fdm[np.ix_(t_idx, x_idx, z_idx)]
    interpolated = resample_wavefield_to_grid(
        sparse_volume,
        source_t=t[t_idx],
        source_x=x[x_idx],
        source_z=z[z_idx],
        target_t=t,
        target_x=x,
        target_z=z,
    )

    metadata = {
        "model_type": "sparse_trilinear_interpolation",
        "time_stride": int(time_stride),
        "space_stride": int(space_stride),
        "num_time_samples": int(len(t_idx)),
        "num_x_samples": int(len(x_idx)),
        "num_z_samples": int(len(z_idx)),
        "num_total_samples": int(len(t_idx) * len(x_idx) * len(z_idx)),
        "sample_fraction": float(len(t_idx) * len(x_idx) * len(z_idx) / u_fdm.size),
        "max_abs_amplitude": float(max_abs_value(interpolated)),
    }
    return interpolated, metadata


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
        loss = cfg.LAMBDA_DATA * loss_data + cfg.REAL_SMOOTHNESS_WEIGHT * loss_pde

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


def _safe_corrcoef(a, b):
    """Return a robust correlation value, or NaN for degenerate arrays."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return np.nan

    a = a[mask]
    b = b[mask]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def read_real_metadata(source_path, section_shape):
    """
    Read useful SEG-Y metadata when available.

    NPY/CSV files are valid reconstruction inputs, but they do not provide the
    geometry needed to make strong field-wave-propagation claims.
    """
    path = Path(source_path)
    suffix = path.suffix.lower()
    metadata = {
        "Source_File": str(path),
        "Source_Format": suffix.replace(".", "") or "unknown",
        "Prepared_Section_Shape": f"{section_shape[0]}x{section_shape[1]}",
        "SEGY_Metadata_Read": False,
        "Sample_Interval_Seconds": np.nan,
        "Trace_Spacing_Units": np.nan,
        "Original_Num_Traces": np.nan,
        "Original_Num_Samples": np.nan,
        "Geometry_Available": False,
    }

    if suffix not in {".sgy", ".segy"}:
        return metadata

    try:
        import segyio
    except ImportError:
        metadata["Metadata_Warning"] = "segyio is not installed; SEG-Y timing/geometry was not read."
        return metadata

    try:
        with segyio.open(str(path), ignore_geometry=True) as f:
            metadata["SEGY_Metadata_Read"] = True
            metadata["Original_Num_Traces"] = int(len(f.trace))
            metadata["Original_Num_Samples"] = int(len(f.samples))

            interval_us = None
            try:
                interval_us = int(f.bin[segyio.BinField.Interval])
            except Exception:
                interval_us = None

            if not interval_us:
                trace_interval_field = getattr(segyio.TraceField, "TRACE_SAMPLE_INTERVAL", None)
                if trace_interval_field is not None and len(f.trace) > 0:
                    try:
                        interval_us = int(f.header[0][trace_interval_field])
                    except Exception:
                        interval_us = None

            if interval_us and interval_us > 0:
                metadata["Sample_Interval_Seconds"] = float(interval_us) * 1e-6

            coordinate_fields = [
                getattr(segyio.TraceField, name, None)
                for name in ("GroupX", "SourceX", "CDP_X", "INLINE_3D")
            ]
            coordinate_fields = [field for field in coordinate_fields if field is not None]
            trace_limit = min(int(len(f.trace)), 512)

            for field in coordinate_fields:
                coords = []
                for i in range(trace_limit):
                    try:
                        coords.append(float(f.header[i][field]))
                    except Exception:
                        continue

                coords = np.asarray(coords, dtype=np.float64)
                coords = coords[np.isfinite(coords)]
                coords = coords[np.abs(coords) > 0.0]
                if coords.size < 3:
                    continue

                diffs = np.diff(np.sort(np.unique(coords)))
                diffs = diffs[np.abs(diffs) > 0.0]
                if diffs.size > 0:
                    metadata["Trace_Spacing_Units"] = float(np.median(np.abs(diffs)))
                    metadata["Geometry_Available"] = True
                    break
    except Exception as exc:
        metadata["Metadata_Warning"] = f"SEG-Y metadata read failed: {exc}"

    return metadata


def section_physical_diagnostics(section, prefix):
    """Compute texture/spectrum diagnostics for a seismic section."""
    arr = np.asarray(section, dtype=np.float64)
    arr = np.nan_to_num(arr)
    centered = arr - np.mean(arr)
    eps = 1e-12

    grad_x, grad_z = np.gradient(centered)
    laplacian = np.gradient(grad_x, axis=0) + np.gradient(grad_z, axis=1)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    energy = float(np.mean(centered ** 2))

    spectrum = np.abs(np.fft.rfft2(centered)) ** 2
    total_power = float(np.sum(spectrum))
    kx = np.fft.fftfreq(arr.shape[0])
    kz = np.fft.rfftfreq(arr.shape[1])
    KX, KZ = np.meshgrid(kx, kz, indexing="ij")
    radial_frequency = np.sqrt(KX ** 2 + KZ ** 2)

    if total_power > eps:
        dominant_frequency = float(radial_frequency.reshape(-1)[int(np.argmax(spectrum.reshape(-1)))])
        spectral_centroid = float(np.sum(radial_frequency * spectrum) / total_power)
        high_frequency_cutoff = 0.35 * float(np.max(radial_frequency))
        high_frequency_fraction = float(np.sum(spectrum[radial_frequency >= high_frequency_cutoff]) / total_power)
    else:
        dominant_frequency = np.nan
        spectral_centroid = np.nan
        high_frequency_fraction = np.nan

    return {
        f"{prefix}_RMS": rms,
        f"{prefix}_Gradient_Energy_Ratio": float(np.mean(grad_x ** 2 + grad_z ** 2) / (energy + eps)),
        f"{prefix}_Laplacian_Roughness_To_RMS": float(np.sqrt(np.mean(laplacian ** 2)) / (rms + eps)),
        f"{prefix}_Trace_Neighbor_Correlation": _safe_corrcoef(arr[:-1, :], arr[1:, :]),
        f"{prefix}_Sample_Neighbor_Correlation": _safe_corrcoef(arr[:, :-1], arr[:, 1:]),
        f"{prefix}_Dominant_Radial_Frequency": dominant_frequency,
        f"{prefix}_Spectral_Centroid": spectral_centroid,
        f"{prefix}_High_Frequency_Energy_Fraction": high_frequency_fraction,
    }


def plot_real_validation_spectrum(section, pred, path):
    """Save a spectral diagnostic figure for the real-data validation section."""
    import matplotlib.pyplot as plt

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    def log_spectrum(arr):
        centered = np.asarray(arr, dtype=np.float64) - np.mean(arr)
        spectrum = np.fft.fftshift(np.abs(np.fft.fft2(centered)) ** 2)
        return np.log1p(spectrum)

    reference_spectrum = log_spectrum(section)
    prediction_spectrum = log_spectrum(pred)
    error_spectrum = log_spectrum(np.asarray(pred) - np.asarray(section))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    items = [
        (reference_spectrum, "Reference spectrum"),
        (prediction_spectrum, "PINN spectrum"),
        (error_spectrum, "Error spectrum"),
    ]
    for ax, (image, title) in zip(axes, items):
        im = ax.imshow(image, cmap="magma", aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_real_physical_validation(source_path, section, pred):
    """Build a one-row validation table clarifying what real-data evidence supports."""
    metadata = read_real_metadata(source_path, section.shape)
    reconstruction_metrics = compute_all_metrics(pred, section)
    section_diag = section_physical_diagnostics(section, "Reference")
    pred_diag = section_physical_diagnostics(pred, "Prediction")

    ref_grad_x, ref_grad_z = np.gradient(section)
    pred_grad_x, pred_grad_z = np.gradient(pred)
    has_timing = np.isfinite(metadata["Sample_Interval_Seconds"])
    has_spacing = np.isfinite(metadata["Trace_Spacing_Units"])

    if has_timing and has_spacing:
        validation_level = "metadata_supported_reconstruction"
        limitation = "Timing and trace geometry are available, but velocity model and source wavelet are still missing."
    else:
        validation_level = "normalized_section_reconstruction_only"
        limitation = "Timing, geometry, velocity model, and source wavelet are not fully available."

    row = {
        "mode": "real_physical_validation",
        **metadata,
        "Validation_Level": validation_level,
        "Can_Claim_Real_Wave_Propagation": False,
        "Claim_Limitation": limitation,
        "Gradient_Field_Correlation": _safe_corrcoef(
            np.stack([ref_grad_x, ref_grad_z], axis=0),
            np.stack([pred_grad_x, pred_grad_z], axis=0),
        ),
        "Amplitude_Error_RMS": float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(section)) ** 2))),
        **reconstruction_metrics,
        **section_diag,
        **pred_diag,
    }
    return pd.DataFrame([row])


def write_real_validation_report(validation_df):
    row = validation_df.iloc[0].to_dict()
    lines = [
        "# Real-Data Physical Validation",
        "",
        "This file separates what the real-data experiment proves from what it does not prove.",
        "",
        f"- Validation level: {row['Validation_Level']}",
        f"- Can claim real seismic wave propagation from this section alone: {row['Can_Claim_Real_Wave_Propagation']}",
        f"- Limitation: {row['Claim_Limitation']}",
        f"- Reconstruction Relative L2 Error: {row.get('Relative_L2_Error', np.nan):.6f}",
        f"- Reconstruction Correlation: {row.get('Correlation', np.nan):.6f}",
        f"- Gradient Field Correlation: {row.get('Gradient_Field_Correlation', np.nan):.6f}",
        "",
        "Defense wording:",
        "",
        "Use the real section as an applied reconstruction and regularization demonstration. Use the controlled synthetic FDM experiments as the main physical proof of acoustic wave propagation in a known heterogeneous medium.",
    ]
    with open("results/data/real_physical_validation_report.md", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


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
    validation_df = build_real_physical_validation(source_path, section, pred)
    validation_df.to_csv("results/data/real_physical_validation.csv", index=False)
    write_real_validation_report(validation_df)
    plot_real_validation_spectrum(section, pred, "results/figures/real_validation_spectrum.png")

    validation_row = validation_df.iloc[0]
    df = pd.DataFrame([
        {
            "mode": "real",
            "source_file": str(source_path),
            "Validation_Level": validation_row["Validation_Level"],
            "Can_Claim_Real_Wave_Propagation": validation_row["Can_Claim_Real_Wave_Propagation"],
            "Gradient_Field_Correlation": validation_row["Gradient_Field_Correlation"],
            **real_metrics,
            "training_time_seconds": training_time,
        }
    ])

    df.to_csv("results/data/real_metrics.csv", index=False)
    df.to_csv("results/data/real_summary.csv", index=False)

    print("\nFinal real-data quantitative results:")
    print(df)

    return df


def run_synthetic_mode(
    run_ablation=False,
    scenario="heterogeneous",
    source_case="single",
    run_multi_seed=False,
):
    source_case = resolve_source_case(source_case)
    active_sources = get_sources_for_case(source_case)
    output_key = artifact_key(scenario, source_case)

    print(f"Step 1: Running finite difference simulation for scenario: {scenario}, source_case: {source_case}")
    start_fdm = time.time()
    result = solve_wave_equation_fdm(model_type=scenario, sources=active_sources, return_metadata=True)
    fdm_time = time.time() - start_fdm
    x, z, t, c, u_fdm, fdm_metadata = result
    fdm_metadata["scenario"] = scenario
    fdm_metadata["source_case"] = source_case

    write_advanced_scope_report()
    velocity_diagnostic = apparent_velocity_diagnostic(x, z, t, u_fdm, sources=active_sources)
    velocity_diagnostic["scenario"] = scenario
    velocity_diagnostic["source_case"] = source_case
    velocity_diagnostic.to_csv(data_path("velocity_inversion_diagnostic.csv", scenario=output_key), index=False)

    print(f"FDM simulation completed in {fdm_time:.2f} seconds")
    print(f"FDM amplitude scale: {fdm_metadata['max_abs_amplitude']:.6e}")
    print(f"FDM CFL number: {fdm_metadata['cfl']:.6f}")
    print(f"Reference wavefield max amplitude: {max_abs_value(u_fdm):.6e}")

    print("Step 2: Running traditional finite difference baselines...")
    start_baseline = time.time()
    baseline_result = solve_wave_equation_fdm(model_type="homogeneous", sources=active_sources, return_metadata=True)
    baseline_time = time.time() - start_baseline
    _, _, _, c_homogeneous, u_homogeneous, baseline_metadata = baseline_result

    print(f"Homogeneous baseline completed in {baseline_time:.2f} seconds")
    print(f"Homogeneous baseline max amplitude: {baseline_metadata['max_abs_amplitude']:.6e}")

    start_layered = time.time()
    layered_result = solve_wave_equation_fdm(model_type="layered", sources=active_sources, return_metadata=True)
    layered_time = time.time() - start_layered
    _, _, _, c_layered, u_layered, layered_metadata = layered_result

    print(f"Layered baseline completed in {layered_time:.2f} seconds")
    print(f"Layered baseline max amplitude: {layered_metadata['max_abs_amplitude']:.6e}")

    start_coarse = time.time()
    coarse_result = solve_wave_equation_fdm(
        model_type=scenario,
        nx=cfg.COARSE_NX,
        nz=cfg.COARSE_NZ,
        nt=cfg.COARSE_NT,
        sources=active_sources,
        return_metadata=True,
    )
    coarse_solve_time = time.time() - start_coarse
    x_coarse, z_coarse, t_coarse, c_coarse, u_coarse, coarse_metadata = coarse_result
    coarse_metadata["scenario"] = scenario

    start_interpolation = time.time()
    u_coarse_resampled = resample_wavefield_to_grid(
        u_coarse,
        source_t=t_coarse,
        source_x=x_coarse,
        source_z=z_coarse,
        target_t=t,
        target_x=x,
        target_z=z,
    )
    coarse_interpolation_time = time.time() - start_interpolation
    coarse_total_time = time.time() - start_coarse
    coarse_metadata["solve_time_seconds"] = coarse_solve_time
    coarse_metadata["interpolation_time_seconds"] = coarse_interpolation_time

    print(f"Coarse-grid heterogeneous FDM baseline completed in {coarse_total_time:.2f} seconds")
    print(f"Coarse-grid baseline max amplitude after interpolation: {max_abs_value(u_coarse_resampled):.6e}")

    start_sparse = time.time()
    u_sparse_interpolated, sparse_metadata = build_sparse_interpolation_baseline(
        u_fdm,
        x=x,
        z=z,
        t=t,
        time_stride=cfg.SPARSE_BASELINE_TIME_STRIDE,
        space_stride=cfg.SPARSE_BASELINE_SPACE_STRIDE,
    )
    sparse_time = time.time() - start_sparse
    sparse_metadata["interpolation_time_seconds"] = sparse_time
    sparse_metadata["scenario"] = scenario
    baseline_metadata["scenario"] = scenario
    layered_metadata["scenario"] = scenario
    sparse_metadata["source_case"] = source_case
    baseline_metadata["source_case"] = source_case
    layered_metadata["source_case"] = source_case
    coarse_metadata["source_case"] = source_case

    print(f"Sparse trilinear interpolation baseline completed in {sparse_time:.2f} seconds")
    print(
        "Sparse interpolation sample fraction: "
        f"{100.0 * sparse_metadata['sample_fraction']:.2f}%"
    )

    pd.DataFrame([
        fdm_metadata,
        baseline_metadata,
        layered_metadata,
        coarse_metadata,
        sparse_metadata,
    ]).to_csv(data_path("fdm_metadata.csv", scenario=output_key), index=False)

    plot_velocity_model(
        c,
        path=figure_path("velocity_model.png", scenario=output_key),
        title=f"{scenario_label(scenario)} Velocity Model",
    )
    plot_velocity_model(
        c_homogeneous,
        path=figure_path("homogeneous_velocity_model.png", scenario=output_key),
    )
    plot_velocity_model(
        c_layered,
        path=figure_path("layered_velocity_model.png", scenario=output_key),
        title="Layered Baseline Velocity Model",
    )
    plot_source_wavelet(t, path=figure_path("ricker_wavelet.png", scenario=output_key), sources=active_sources)

    snapshot_indices = get_snapshot_indices(t)

    for idx in snapshot_indices:
        plot_wave_snapshot(
            u_fdm,
            idx,
            title=f"FDM Wavefield Snapshot at t={t[idx]:.3f}",
            path=figure_path(f"fdm_snapshot_{idx}.png", scenario=output_key),
        )
        plot_baseline_comparison(
            reference=u_fdm,
            baseline=u_homogeneous,
            time_index=idx,
            path=figure_path(f"baseline_comparison_t_{idx}.png", scenario=output_key),
            baseline_title="Homogeneous FDM Baseline",
        )
        plot_baseline_comparison(
            reference=u_fdm,
            baseline=u_layered,
            time_index=idx,
            path=figure_path(f"layered_baseline_comparison_t_{idx}.png", scenario=output_key),
            baseline_title="Layered FDM Baseline",
        )
        plot_baseline_comparison(
            reference=u_fdm,
            baseline=u_coarse_resampled,
            time_index=idx,
            path=figure_path(f"coarse_baseline_comparison_t_{idx}.png", scenario=output_key),
            baseline_title="Coarse Heterogeneous FDM Baseline",
        )
        plot_baseline_comparison(
            reference=u_fdm,
            baseline=u_sparse_interpolated,
            time_index=idx,
            path=figure_path(f"sparse_interpolation_comparison_t_{idx}.png", scenario=output_key),
            baseline_title="Sparse Trilinear Interpolation Baseline",
        )

    base_neural_model_name = "data_only_nn" if cfg.LAMBDA_PDE == 0 else "weak_pde_pinn"
    neural_model_name = f"{base_neural_model_name}{scenario_suffix(output_key)}"
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
        velocity_model_type=scenario,
        source_locations=active_sources,
    )
    pinn_training_time = time.time() - start_pinn
    print(f"Neural model training completed in {pinn_training_time:.2f} seconds")

    plot_loss_curve(loss_history, path=figure_path("loss_curve.png", scenario=output_key))
    save_loss_history(loss_history, data_path("loss_history.csv", scenario=output_key))

    residual_summary = pd.DataFrame([
        {
            "mode": neural_model_name,
            "scenario": scenario,
            "source_case": source_case,
            **evaluate_pde_residual(model),
        }
    ])
    residual_summary.to_csv(data_path("pde_residual_summary.csv", scenario=output_key), index=False)

    metrics_rows = []
    homogeneous_baseline_rows = []
    layered_baseline_rows = []
    coarse_baseline_rows = []
    sparse_baseline_rows = []

    for idx in snapshot_indices:
        pinn_snapshot = predict_snapshot(model, time_value=t[idx])
        snapshot_metrics = compute_all_metrics(pinn_snapshot, u_fdm[idx])
        homogeneous_baseline_metrics = compute_all_metrics(u_homogeneous[idx], u_fdm[idx])
        layered_baseline_metrics = compute_all_metrics(u_layered[idx], u_fdm[idx])
        coarse_baseline_metrics = compute_all_metrics(u_coarse_resampled[idx], u_fdm[idx])
        sparse_baseline_metrics = compute_all_metrics(u_sparse_interpolated[idx], u_fdm[idx])

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

        coarse_baseline_rows.append(
            {
                "time_index": idx,
                "time": float(t[idx]),
                **coarse_baseline_metrics,
            }
        )

        sparse_baseline_rows.append(
            {
                "time_index": idx,
                "time": float(t[idx]),
                **sparse_baseline_metrics,
            }
        )

        plot_comparison(
            fdm=u_fdm,
            pinn=pinn_snapshot,
            time_index=idx,
            path=figure_path(f"comparison_t_{idx}.png", scenario=output_key),
        )

    df = pd.DataFrame(metrics_rows)
    homogeneous_baseline_df = pd.DataFrame(homogeneous_baseline_rows)
    layered_baseline_df = pd.DataFrame(layered_baseline_rows)
    coarse_baseline_df = pd.DataFrame(coarse_baseline_rows)
    sparse_baseline_df = pd.DataFrame(sparse_baseline_rows)

    df.to_csv(data_path("metrics.csv", scenario=output_key), index=False)
    homogeneous_baseline_df.to_csv(data_path("homogeneous_baseline_metrics.csv", scenario=output_key), index=False)
    layered_baseline_df.to_csv(data_path("layered_baseline_metrics.csv", scenario=output_key), index=False)
    coarse_baseline_df.to_csv(data_path("coarse_baseline_metrics.csv", scenario=output_key), index=False)
    sparse_baseline_df.to_csv(data_path("sparse_interpolation_baseline_metrics.csv", scenario=output_key), index=False)

    # Backward-compatible filename used by earlier report versions.
    homogeneous_baseline_df.to_csv(data_path("baseline_metrics.csv", scenario=output_key), index=False)

    neural_comparison_row = summarize_metric_table(
        df,
        model_name=neural_display_name,
        role="neural_model",
        training_time_seconds=pinn_training_time,
    )
    neural_comparison_row.update(residual_summary.drop(columns=["mode"]).iloc[0].to_dict())

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
            coarse_baseline_df,
            model_name="Coarse heterogeneous FDM baseline",
            role="traditional_coarse_grid_baseline",
            training_time_seconds=coarse_total_time,
        ),
        summarize_metric_table(
            sparse_baseline_df,
            model_name="Sparse trilinear interpolation baseline",
            role="traditional_sparse_interpolation_baseline",
            training_time_seconds=sparse_time,
        ),
        neural_comparison_row,
    ]

    for row in comparison_rows:
        row["scenario"] = scenario
        row["source_case"] = source_case

    ablation_summary = None
    if run_ablation:
        ablation_summary = run_ablation_study(
            x,
            z,
            t,
            u_fdm,
            snapshot_indices,
            scenario=scenario,
            source_case=source_case,
            source_locations=active_sources,
        )
        if ablation_summary is not None and not ablation_summary.empty:
            comparison_rows.extend(ablation_summary.to_dict("records"))

    multi_seed_summary = None
    if run_multi_seed:
        multi_seed_summary = run_multi_seed_repeats(
            x,
            z,
            t,
            u_fdm,
            snapshot_indices,
            scenario=scenario,
            source_case=source_case,
            source_locations=active_sources,
        )
        if multi_seed_summary is not None and not multi_seed_summary.empty:
            comparison_rows.extend(multi_seed_summary.to_dict("records"))

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(data_path("model_comparison_summary.csv", scenario=output_key), index=False)
    save_model_comparison_figures(comparison_df, scenario=output_key)

    summary_df = pd.DataFrame([
        {
            "mode": "synthetic",
            "scenario": scenario,
            "source_case": source_case,
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
            "Coarse_Mean_Relative_L2_Error": coarse_baseline_df["Relative_L2_Error"].mean(),
            "Coarse_Mean_NRMSE": coarse_baseline_df["NRMSE"].mean(),
            "Coarse_Mean_Correlation": coarse_baseline_df["Correlation"].mean(),
            "SparseInterp_Mean_Relative_L2_Error": sparse_baseline_df["Relative_L2_Error"].mean(),
            "SparseInterp_Mean_NRMSE": sparse_baseline_df["NRMSE"].mean(),
            "SparseInterp_Mean_Correlation": sparse_baseline_df["Correlation"].mean(),
            "SparseInterp_sample_fraction": sparse_metadata["sample_fraction"],
            "FDM_time_seconds": fdm_time,
            "Homogeneous_FDM_time_seconds": baseline_time,
            "Layered_FDM_time_seconds": layered_time,
            "Coarse_FDM_time_seconds": coarse_total_time,
            "SparseInterp_time_seconds": sparse_time,
            "VelocityDiagnostic_apparent_velocity": (
                velocity_diagnostic["apparent_velocity"].dropna().mean()
                if "apparent_velocity" in velocity_diagnostic.columns
                else np.nan
            ),
            "Neural_model_name": neural_model_name,
            "Neural_display_name": neural_display_name,
            "PINN_training_time_seconds": pinn_training_time,
        }
    ])

    summary_df.to_csv(data_path("synthetic_summary.csv", scenario=output_key), index=False)

    print("\nFinal synthetic quantitative results:")
    print(df)

    print("\nHomogeneous baseline quantitative results:")
    print(homogeneous_baseline_df)

    print("\nLayered baseline quantitative results:")
    print(layered_baseline_df)

    print("\nCoarse-grid heterogeneous baseline quantitative results:")
    print(coarse_baseline_df)

    print("\nSparse trilinear interpolation baseline quantitative results:")
    print(sparse_baseline_df)

    print("\nSynthetic summary:")
    print(summary_df)

    print("\nPDE residual diagnostics:")
    print(residual_summary)

    print("\nModel comparison summary:")
    print(comparison_df)

    return df, summary_df


def run_ablation_study(
    x,
    z,
    t,
    u_fdm,
    snapshot_indices,
    scenario="heterogeneous",
    source_case="single",
    source_locations=None,
):
    """Run optional smaller ablation experiments for dissertation comparison tables."""
    output_key = artifact_key(scenario, source_case)
    print(f"\nRunning optional ablation study for scenario: {scenario}, source_case: {source_case}")
    ablation_epochs = int(getattr(cfg, "ABLATION_EPOCHS", max(500, cfg.EPOCHS // 2)))
    experiments = [
        {
            "experiment": "data_only_fourier",
            "model": "Data-only NN with Fourier",
            "role": "ablation_remove_physics",
            "use_fourier": True,
            "lambda_pde": 0.0,
            "factor_tested": "PDE term removed",
        },
        {
            "experiment": "data_only_no_fourier",
            "model": "Data-only NN without Fourier",
            "role": "ablation_remove_physics_and_fourier",
            "use_fourier": False,
            "lambda_pde": 0.0,
            "factor_tested": "PDE and Fourier terms removed",
        },
        {
            "experiment": "weak_pde_no_fourier",
            "model": "Weak-PDE PINN without Fourier",
            "role": "ablation_remove_fourier",
            "use_fourier": False,
            "lambda_pde": cfg.LAMBDA_PDE,
            "factor_tested": "Fourier features removed",
        },
        {
            "experiment": "stronger_pde_fourier",
            "model": "Stronger-PDE PINN with Fourier",
            "role": "ablation_pde_weight_sensitivity",
            "use_fourier": True,
            "lambda_pde": 1e-7,
            "factor_tested": "PDE weight increased",
        },
    ]

    rows = []
    for exp in experiments:
        experiment_name = f"{exp['experiment']}{scenario_suffix(output_key)}"
        print(f"\nAblation experiment: {experiment_name}")
        start = time.time()
        model, loss_history = train_pinn(
            x,
            z,
            t,
            u_fdm,
            experiment_name=experiment_name,
            use_fourier=exp["use_fourier"],
            lambda_pde=exp["lambda_pde"],
            epochs=ablation_epochs,
            velocity_model_type=scenario,
            source_locations=source_locations,
        )
        training_time = time.time() - start
        save_loss_history(loss_history, data_path(f"loss_history_{exp['experiment']}.csv", scenario=output_key))

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
        row["experiment"] = exp["experiment"]
        row["scenario"] = scenario
        row["source_case"] = source_case
        row["factor_tested"] = exp["factor_tested"]
        row["ablation_epochs"] = ablation_epochs
        row["lambda_pde"] = exp["lambda_pde"]
        row["use_fourier"] = exp["use_fourier"]
        row.update(evaluate_pde_residual(model))
        rows.append(row)

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(data_path("ablation_summary.csv", scenario=output_key), index=False)
    return ablation_df


def run_multi_seed_repeats(
    x,
    z,
    t,
    u_fdm,
    snapshot_indices,
    scenario="heterogeneous",
    source_case="single",
    source_locations=None,
):
    """Run shorter repeated trainings to quantify seed sensitivity."""
    output_key = artifact_key(scenario, source_case)
    seeds = [int(seed) for seed in getattr(cfg, "MULTI_SEED_VALUES", [cfg.RANDOM_SEED])]
    repeat_epochs = int(getattr(cfg, "MULTI_SEED_EPOCHS", max(500, cfg.EPOCHS // 2)))

    print(
        f"\nRunning multi-seed repeats for scenario={scenario}, "
        f"source_case={source_case}: seeds={seeds}, epochs={repeat_epochs}"
    )
    rows = []
    for seed in seeds:
        experiment_name = f"multi_seed_{seed}{scenario_suffix(output_key)}"
        print(f"\nMulti-seed repeat: seed={seed}, experiment={experiment_name}")
        start = time.time()
        model, loss_history = train_pinn(
            x,
            z,
            t,
            u_fdm,
            experiment_name=experiment_name,
            lambda_pde=cfg.LAMBDA_PDE,
            epochs=repeat_epochs,
            seed=seed,
            velocity_model_type=scenario,
            source_locations=source_locations,
        )
        training_time = time.time() - start
        save_loss_history(loss_history, data_path(f"loss_history_multi_seed_{seed}.csv", scenario=output_key))

        metric_rows = []
        for idx in snapshot_indices:
            pred = predict_snapshot(model, time_value=t[idx])
            metric_rows.append(compute_all_metrics(pred, u_fdm[idx]))

        metric_df = pd.DataFrame(metric_rows)
        row = summarize_metric_table(
            metric_df,
            model_name=f"Weak-PDE PINN seed {seed}",
            role="multi_seed_repeat",
            training_time_seconds=training_time,
        )
        row["experiment"] = "multi_seed_repeat"
        row["scenario"] = scenario
        row["source_case"] = source_case
        row["seed"] = seed
        row["repeat_epochs"] = repeat_epochs
        row.update(evaluate_pde_residual(model))
        rows.append(row)

    repeat_df = pd.DataFrame(rows)
    repeat_df.to_csv(data_path("multi_seed_summary.csv", scenario=output_key), index=False)

    numeric_metrics = [
        "Mean_MSE",
        "Mean_MAE",
        "Mean_Relative_L2_Error",
        "Mean_NRMSE",
        "Mean_PSNR_dB",
        "Mean_Correlation",
        "Mean_Energy_Ratio",
        "Residual_MAE",
        "Residual_RMSE",
    ]
    aggregate_rows = []
    for metric in numeric_metrics:
        if metric not in repeat_df.columns:
            continue
        values = pd.to_numeric(repeat_df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        aggregate_rows.append(
            {
                "scenario": scenario,
                "source_case": source_case,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "num_seeds": int(len(values)),
                "repeat_epochs": repeat_epochs,
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(data_path("multi_seed_aggregate.csv", scenario=output_key), index=False)

    print("\nMulti-seed summary:")
    print(repeat_df)
    print("\nMulti-seed aggregate:")
    print(aggregate_df)

    return repeat_df


def run_synthetic_scenarios(
    run_ablation=False,
    scenario_arg="heterogeneous",
    source_case="single",
    run_multi_seed=False,
):
    """Run one or all configured synthetic scenarios and save combined tables."""
    scenarios = resolve_scenarios(scenario_arg)
    source_case = resolve_source_case(source_case)
    all_metrics = []
    all_summaries = []
    all_comparisons = []

    for scenario in scenarios:
        metrics_df, summary_df = run_synthetic_mode(
            run_ablation=run_ablation,
            scenario=scenario,
            source_case=source_case,
            run_multi_seed=run_multi_seed,
        )
        all_metrics.append(metrics_df.assign(scenario=scenario, source_case=source_case))
        all_summaries.append(summary_df)

        comparison_file = data_path("model_comparison_summary.csv", scenario=artifact_key(scenario, source_case))
        if os.path.exists(comparison_file):
            all_comparisons.append(pd.read_csv(comparison_file))

    combined_metrics = pd.concat(all_metrics, ignore_index=True, sort=False)
    combined_summary = pd.concat(all_summaries, ignore_index=True, sort=False)

    if len(scenarios) > 1 or source_case != "single":
        suffix = source_case_suffix(source_case)
        combined_metrics.to_csv(f"results/data/metrics_scenario_summary{suffix}.csv", index=False)
        combined_summary.to_csv(f"results/data/synthetic_scenario_summary{suffix}.csv", index=False)
        if all_comparisons:
            pd.concat(all_comparisons, ignore_index=True, sort=False).to_csv(
                f"results/data/model_comparison_scenario_summary{suffix}.csv",
                index=False,
            )

    return combined_metrics, combined_summary


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
        "The robustness extension includes a second faulted heterogeneous velocity model and optional multi-seed repeats. These additions help show that the result is not tied to one easy velocity model or one lucky random initialization.",
        "",
        "The advanced-case extension implements the safe high-value additions: multiple seismic sources and an apparent-velocity diagnostic. Full 3D, elastic P/S-wave, and anisotropic PINNs are kept as future work because they require different governing equations, baselines, and validation data.",
        "",
        "## Secondary real-data demonstration",
        "",
        "The real seismic section is included as an applied demonstration. It shows that the neural representation can reconstruct a real seismic section, but it should be described as real-data reconstruction with physics-inspired smoothness regularization unless the exact field velocity model, acquisition geometry, and source wavelet are available. The file results/data/real_physical_validation.csv records this claim level explicitly.",
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
        "- results/figures/coarse_baseline_comparison_t_*.png",
        "- results/figures/sparse_interpolation_comparison_t_*.png",
        "- results/figures/layered_velocity_model.png",
        "- results/figures/ricker_wavelet.png",
        "- results/figures/comparison_t_*.png",
        "- results/figures/loss_curve.png",
        "- results/data/metrics.csv",
        "- results/data/synthetic_summary.csv",
        "- results/data/homogeneous_baseline_metrics.csv",
        "- results/data/layered_baseline_metrics.csv",
        "- results/data/coarse_baseline_metrics.csv",
        "- results/data/sparse_interpolation_baseline_metrics.csv",
        "- results/data/pde_residual_summary.csv",
        "- results/data/fdm_metadata.csv",
        "- results/data/model_comparison_summary.csv",
        "- results/data/advanced_scope_matrix.csv",
        "- results/data/advanced_scope_report.md",
        "- results/data/velocity_inversion_diagnostic.csv",
        "- results/figures/comparison_relative_l2_bar.png",
        "- results/figures/comparison_correlation_bar.png",
        "- results/figures/comparison_energy_ratio_bar.png",
        "- results/data/ablation_summary.csv, if ablation mode was executed",
        "- results/data/synthetic_scenario_summary.csv, if --scenario all was executed",
        "- results/data/model_comparison_scenario_summary.csv, if --scenario all was executed",
        "- results/data/*_multi_source.csv, if --source-case multi_source was executed",
        "- results/data/multi_seed_summary.csv, if --multi-seed was executed",
        "- results/data/multi_seed_aggregate.csv, if --multi-seed was executed",
        "- results/data/real_metrics.csv, if real mode was executed",
        "- results/data/real_physical_validation.csv, if real mode was executed",
        "- results/data/real_physical_validation_report.md, if real mode was executed",
        "- results/figures/real_validation_spectrum.png, if real mode was executed",
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
            "",
            "## Added baseline validation",
            "",
            "Homogeneous, layered, coarse-grid FDM, and sparse trilinear interpolation baselines are included as traditional baselines. This allows the defense to compare the neural result against simplified physical models, a cheaper numerical approximation, a non-neural reconstruction method, and the heterogeneous reference solution with anomalies.",
            "",
        ]
    except ValueError:
        pass
    with open("results/data/defense_report.md", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def run_all_mode(
    run_ablation=False,
    scenario_arg="heterogeneous",
    source_case="single",
    run_multi_seed=False,
):
    print("Running full defense pipeline: synthetic experiment + real-data reconstruction if available.")
    synthetic_metrics, synthetic_summary = run_synthetic_scenarios(
        run_ablation=run_ablation,
        scenario_arg=scenario_arg,
        source_case=source_case,
        run_multi_seed=run_multi_seed,
    )

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
    parser.add_argument(
        "--scenario",
        choices=cfg.SYNTHETIC_SCENARIOS + ["all"],
        default="heterogeneous",
        help="Synthetic velocity scenario to run. Use 'all' for robustness validation.",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Run shorter repeated PINN trainings with multiple random seeds.",
    )
    parser.add_argument(
        "--source-case",
        choices=["single", "multi_source"],
        default="single",
        help="Use the default single source or the optional multi-source shot case.",
    )
    args = parser.parse_args()

    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)

    print_config(
        args.mode,
        scenario=args.scenario,
        source_case=args.source_case,
        run_multi_seed=args.multi_seed,
    )
    save_config_snapshot(
        args.mode,
        scenario=args.scenario,
        source_case=args.source_case,
        run_multi_seed=args.multi_seed,
    )

    if args.mode == "real":
        real_summary = run_real_mode()
        write_defense_report(real_summary=real_summary)
    elif args.mode == "synthetic":
        _, synthetic_summary = run_synthetic_scenarios(
            run_ablation=args.ablation,
            scenario_arg=args.scenario,
            source_case=args.source_case,
            run_multi_seed=args.multi_seed,
        )
        synthetic_summary.to_csv("results/data/summary.csv", index=False)
        write_defense_report(synthetic_summary=synthetic_summary)
    else:
        run_all_mode(
            run_ablation=args.ablation,
            scenario_arg=args.scenario,
            source_case=args.source_case,
            run_multi_seed=args.multi_seed,
        )

    write_thesis_assets()

    print("\nProject completed.")
    print("Generated figures are saved in: results/figures")
    print("Results are saved in: results/data")


if __name__ == "__main__":
    main()
