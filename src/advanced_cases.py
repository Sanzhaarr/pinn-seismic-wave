from pathlib import Path

import numpy as np
import pandas as pd


def advanced_scope_table():
    """Return a dissertation-scope matrix for advanced seismic extensions."""
    rows = [
        {
            "case": "Multiple seismic sources",
            "recommendation": "implement_now",
            "status": "implemented_optional_case",
            "why": "Same 2D acoustic equation, higher wavefield complexity, strong dissertation value.",
            "dissertation_use": "Robustness experiment: interacting wavefronts from several shots.",
        },
        {
            "case": "Velocity inversion for unknown velocity",
            "recommendation": "diagnostic_now_full_method_future",
            "status": "implemented_apparent_velocity_diagnostic",
            "why": "Full inversion is a separate research problem, but an apparent-velocity diagnostic is useful and honest.",
            "dissertation_use": "Discuss as feasibility evidence and future work, not as full inversion.",
        },
        {
            "case": "3D wave propagation",
            "recommendation": "future_work",
            "status": "not_in_main_claim",
            "why": "Needs much larger memory, runtime, and 3D visualization; full validation would dominate the thesis.",
            "dissertation_use": "Mention as natural extension after 2D validation.",
        },
        {
            "case": "Elastic P-wave/S-wave equations",
            "recommendation": "future_work",
            "status": "not_in_main_claim",
            "why": "Requires vector displacement, Lamé parameters, P/S mode conversion, and new baselines.",
            "dissertation_use": "Mention as physics extension beyond acoustic approximation.",
        },
        {
            "case": "Anisotropic media",
            "recommendation": "future_work",
            "status": "not_in_main_claim",
            "why": "Requires anisotropic stiffness/velocity tensors and a different residual formulation.",
            "dissertation_use": "Mention as realistic subsurface extension after isotropic heterogeneous media.",
        },
    ]
    return pd.DataFrame(rows)


def write_advanced_scope_report(
    csv_path="results/data/advanced_scope_matrix.csv",
    markdown_path="results/data/advanced_scope_report.md",
):
    """Save a concise report explaining which advanced cases are in scope."""
    table = advanced_scope_table()
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)

    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in table.columns) + " |"
        for _, row in table.iterrows()
    ]

    lines = [
        "# Advanced Case Scope",
        "",
        "This project deliberately keeps the main scientific claim on 2D acoustic wave propagation in heterogeneous media. Some advanced cases are useful as optional robustness checks, while others are better presented as future work.",
        "",
        header,
        separator,
        *body,
        "",
        "Recommended defense position:",
        "",
        "Multiple sources and apparent-velocity diagnostics strengthen the current dissertation without changing the main equation. Full 3D, elastic, and anisotropic PINNs should be described as future work because each requires a substantially different physical model and validation pipeline.",
    ]
    Path(markdown_path).write_text("\n".join(lines), encoding="utf-8")
    return table


def apparent_velocity_diagnostic(x, z, t, u, sources=None, quantile=0.90):
    """
    Estimate apparent propagation velocity from energy-radius growth.

    This is not full velocity inversion. It is a lightweight diagnostic showing
    whether the synthetic wavefield contains physically plausible radial energy
    propagation from the configured source positions.
    """
    if sources is None:
        sources = [{"id": "source_0", "x": 0.5, "z": 0.5, "t0": 0.15}]

    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    X, Z = np.meshgrid(x, z, indexing="ij")

    rows = []
    for index, source in enumerate(sources):
        source_id = source.get("id", f"source_{index}")
        source_x = float(source.get("x", 0.5))
        source_z = float(source.get("z", 0.5))
        source_t0 = float(source.get("t0", 0.15))
        radius = np.sqrt((X - source_x) ** 2 + (Z - source_z) ** 2)

        times = []
        radii = []
        for time_index, time_value in enumerate(t):
            if time_value <= source_t0 + 0.03:
                continue

            energy = u[time_index] ** 2
            if not np.any(np.isfinite(energy)) or np.max(energy) <= 1e-18:
                continue

            threshold = np.quantile(energy.reshape(-1), quantile)
            mask = energy >= threshold
            selected_energy = energy[mask]
            if selected_energy.size < 8 or np.sum(selected_energy) <= 1e-18:
                continue

            weighted_radius = float(np.sum(radius[mask] * selected_energy) / np.sum(selected_energy))
            times.append(float(time_value - source_t0))
            radii.append(weighted_radius)

        times = np.asarray(times, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        if times.size >= 3:
            slope, intercept = np.polyfit(times, radii, deg=1)
            fitted = slope * times + intercept
            ss_res = float(np.sum((radii - fitted) ** 2))
            ss_tot = float(np.sum((radii - np.mean(radii)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else np.nan
        else:
            slope = np.nan
            intercept = np.nan
            r2 = np.nan

        rows.append(
            {
                "source_id": source_id,
                "source_x": source_x,
                "source_z": source_z,
                "source_t0": source_t0,
                "apparent_velocity": float(slope) if np.isfinite(slope) else np.nan,
                "fit_intercept": float(intercept) if np.isfinite(intercept) else np.nan,
                "fit_r2": float(r2) if np.isfinite(r2) else np.nan,
                "num_time_samples": int(times.size),
                "diagnostic": "energy_radius_growth",
                "claim_level": "apparent_velocity_only_not_full_inversion",
            }
        )

    return pd.DataFrame(rows)
