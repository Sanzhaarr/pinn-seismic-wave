from pathlib import Path


FORMULAS_MARKDOWN = r"""# Thesis Formulas

Use these equations in the methodology and evaluation chapters.

## Acoustic Wave Equation

The controlled synthetic experiments solve the 2D acoustic wave equation in a heterogeneous medium:

```latex
\frac{\partial^2 u(x,z,t)}{\partial t^2}
= c(x,z)^2
\left(
\frac{\partial^2 u(x,z,t)}{\partial x^2}
+ \frac{\partial^2 u(x,z,t)}{\partial z^2}
\right)
+ s(x,z,t)
```

where \(u(x,z,t)\) is the seismic wavefield, \(c(x,z)\) is the spatially varying velocity model, and \(s(x,z,t)\) is the source term.

## PINN Approximation

The neural network approximates the wavefield as:

```latex
\hat{u}_\theta(x,z,t) \approx u(x,z,t)
```

where \(\theta\) denotes the trainable neural-network parameters.

## PDE Residual

Away from the point-source neighborhood, the physics-informed residual is:

```latex
r_\theta(x,z,t)
=
\frac{\partial^2 \hat{u}_\theta}{\partial t^2}
- c(x,z)^2
\left(
\frac{\partial^2 \hat{u}_\theta}{\partial x^2}
+ \frac{\partial^2 \hat{u}_\theta}{\partial z^2}
\right)
```

The residual is evaluated by automatic differentiation.

## Training Objective

The synthetic PINN is trained with a weighted objective:

```latex
\mathcal{L}
=
\lambda_{\mathrm{data}}\mathcal{L}_{\mathrm{data}}
+ \lambda_{\mathrm{pde}}\mathcal{L}_{\mathrm{pde}}
+ \lambda_{\mathrm{ic}}\mathcal{L}_{\mathrm{ic}}
+ \lambda_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}}
+ \lambda_{\mathrm{amp}}\mathcal{L}_{\mathrm{amp}}
```

with:

```latex
\mathcal{L}_{\mathrm{data}}
=
\frac{1}{N_d}
\sum_{i=1}^{N_d}
\left(
\hat{u}_\theta(x_i,z_i,t_i)-u_{\mathrm{FDM}}(x_i,z_i,t_i)
\right)^2
```

```latex
\mathcal{L}_{\mathrm{pde}}
=
\frac{1}{N_c}
\sum_{j=1}^{N_c}
r_\theta(x_j,z_j,t_j)^2
```

The project uses a small nonzero PDE weight. This makes the model a weak-PDE or PDE-regularized PINN: the data term fits the finite-difference wavefield, while the PDE term regularizes the solution toward acoustic-wave physics.

## Ricker Source Wavelet

The synthetic source is a Ricker wavelet:

```latex
s(t)
=
A
\left(
1 - 2\pi^2 f_0^2 (t-t_0)^2
\right)
\exp\left(
-\pi^2 f_0^2 (t-t_0)^2
\right)
```

For the final multi-source experiment, the total source is a sum of individual Ricker sources:

```latex
s(x,z,t)
=
\sum_{k=1}^{N_s}
A_k
\left(
1 - 2\pi^2 f_k^2 (t-t_{0,k})^2
\right)
\exp\left(
-\pi^2 f_k^2 (t-t_{0,k})^2
\right)
\delta(x-x_k)\delta(z-z_k)
```

## Finite-Difference Update

The reference wavefield is generated with a second-order finite-difference update:

```latex
u_{i,j}^{n+1}
=
2u_{i,j}^{n}
- u_{i,j}^{n-1}
+ \Delta t^2 c_{i,j}^2
\left[
\frac{u_{i+1,j}^{n}-2u_{i,j}^{n}+u_{i-1,j}^{n}}{\Delta x^2}
+
\frac{u_{i,j+1}^{n}-2u_{i,j}^{n}+u_{i,j-1}^{n}}{\Delta z^2}
\right]
+ \Delta t^2 s_{i,j}^{n}
```

## Stability Condition

The CFL stability condition used for the 2D FDM scheme is:

```latex
\mathrm{CFL}
=
c_{\max}\Delta t
\sqrt{
\frac{1}{\Delta x^2}
+
\frac{1}{\Delta z^2}
}
< 1
```

## Evaluation Metrics

Mean squared error:

```latex
\mathrm{MSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{u}_i-u_i)^2
```

Mean absolute error:

```latex
\mathrm{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
|\hat{u}_i-u_i|
```

Relative L2 error:

```latex
\mathrm{Relative\ L2}
=
\frac{\|\hat{u}-u\|_2}{\|u\|_2}
```

Normalized RMSE:

```latex
\mathrm{NRMSE}
=
\frac{
\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{u}_i-u_i)^2}
}{
\max_i |u_i|
}
```

Correlation coefficient:

```latex
\rho
=
\frac{
\sum_i(\hat{u}_i-\bar{\hat{u}})(u_i-\bar{u})
}{
\sqrt{\sum_i(\hat{u}_i-\bar{\hat{u}})^2}
\sqrt{\sum_i(u_i-\bar{u})^2}
}
```

Energy ratio:

```latex
E_{\mathrm{ratio}}
=
\frac{\sum_i \hat{u}_i^2}{\sum_i u_i^2}
```

## Recommended Scope Statement

```text
This dissertation investigates physics-informed neural networks for 2D acoustic seismic wave propagation in heterogeneous media. The main validation is performed on controlled synthetic experiments where the velocity model, source, boundary treatment, and finite-difference reference solution are known. Real seismic data are used as an additional reconstruction demonstration rather than as full proof of physical field-scale wave propagation.
```
"""


FIGURE_CANDIDATES = [
    (
        "velocity_model_heterogeneous_multi_source",
        "results/figures/velocity_model_heterogeneous_multi_source.png",
        "Primary heterogeneous velocity model used in the final multi-source experiment.",
        "Methodology / Synthetic setup",
        1,
    ),
    (
        "velocity_model_faulted_multi_source",
        "results/figures/velocity_model_faulted_multi_source.png",
        "Faulted heterogeneous velocity model used for robustness validation.",
        "Methodology / Synthetic setup",
        1,
    ),
    (
        "ricker_wavelet_multi_source",
        "results/figures/ricker_wavelet_heterogeneous_multi_source.png",
        "Multiple Ricker source wavelets used in the final finite-difference simulation.",
        "Methodology / Source model",
        1,
    ),
    (
        "fdm_snapshot_heterogeneous_middle",
        "results/figures/fdm_snapshot_99_heterogeneous_multi_source.png",
        "Finite-difference reference wavefield for the heterogeneous multi-source case.",
        "Results / Reference simulation",
        1,
    ),
    (
        "fdm_snapshot_faulted_middle",
        "results/figures/fdm_snapshot_99_faulted_multi_source.png",
        "Finite-difference reference wavefield for the faulted multi-source case.",
        "Results / Reference simulation",
        1,
    ),
    (
        "pinn_comparison_heterogeneous_middle",
        "results/figures/comparison_t_99_heterogeneous_multi_source.png",
        "PINN prediction, FDM reference, and error for the heterogeneous multi-source case.",
        "Results / PINN vs FDM",
        1,
    ),
    (
        "pinn_comparison_faulted_middle",
        "results/figures/comparison_t_99_faulted_multi_source.png",
        "PINN prediction, FDM reference, and error for the faulted multi-source case.",
        "Results / PINN vs FDM",
        1,
    ),
    (
        "pinn_comparison_faulted_late",
        "results/figures/comparison_t_198_faulted_multi_source.png",
        "Late-time PINN comparison for the harder faulted multi-source case.",
        "Results / PINN vs FDM",
        1,
    ),
    (
        "homogeneous_baseline_multi_source",
        "results/figures/baseline_comparison_t_99_faulted_multi_source.png",
        "Traditional homogeneous FDM baseline compared with the faulted heterogeneous reference.",
        "Results / Traditional baselines",
        2,
    ),
    (
        "layered_baseline_multi_source",
        "results/figures/layered_baseline_comparison_t_99_faulted_multi_source.png",
        "Traditional layered FDM baseline compared with the faulted heterogeneous reference.",
        "Results / Traditional baselines",
        2,
    ),
    (
        "coarse_baseline_multi_source",
        "results/figures/coarse_baseline_comparison_t_99_faulted_multi_source.png",
        "Coarse-grid heterogeneous FDM baseline interpolated to the reference grid.",
        "Results / Traditional baselines",
        2,
    ),
    (
        "sparse_interpolation_baseline_multi_source",
        "results/figures/sparse_interpolation_comparison_t_99_faulted_multi_source.png",
        "Sparse trilinear interpolation baseline compared with the faulted reference wavefield.",
        "Results / Non-neural reconstruction baseline",
        2,
    ),
    (
        "loss_curve_heterogeneous_multi_source",
        "results/figures/loss_curve_heterogeneous_multi_source.png",
        "PINN training loss components for the heterogeneous multi-source case.",
        "Results / Training dynamics",
        1,
    ),
    (
        "loss_curve_faulted_multi_source",
        "results/figures/loss_curve_faulted_multi_source.png",
        "PINN training loss components for the faulted multi-source case.",
        "Results / Training dynamics",
        2,
    ),
    (
        "relative_l2_bar_heterogeneous_multi_source",
        "results/figures/comparison_relative_l2_bar_heterogeneous_multi_source.png",
        "Mean relative L2 error comparison for the heterogeneous multi-source case.",
        "Results / Quantitative comparison",
        1,
    ),
    (
        "relative_l2_bar_faulted_multi_source",
        "results/figures/comparison_relative_l2_bar_faulted_multi_source.png",
        "Mean relative L2 error comparison for the faulted multi-source case.",
        "Results / Quantitative comparison",
        1,
    ),
    (
        "correlation_bar_faulted_multi_source",
        "results/figures/comparison_correlation_bar_faulted_multi_source.png",
        "Mean correlation comparison for the faulted multi-source case.",
        "Results / Quantitative comparison",
        2,
    ),
    (
        "real_section_reference",
        "results/figures/real_section_reference.png",
        "Normalized real seismic section used for the applied reconstruction demonstration.",
        "Additional experiment / Real data",
        3,
    ),
    (
        "real_section_comparison",
        "results/figures/real_section_comparison.png",
        "Real-section reconstruction comparison using the trained neural representation.",
        "Additional experiment / Real data",
        3,
    ),
    (
        "real_validation_spectrum",
        "results/figures/real_validation_spectrum.png",
        "Spectral diagnostic for the real-data reconstruction demonstration.",
        "Additional experiment / Real data",
        3,
    ),
    (
        "legacy_velocity_model",
        "results/figures/velocity_model.png",
        "Legacy single-source heterogeneous velocity model, retained as backup.",
        "Appendix / Backup figures",
        4,
    ),
    (
        "legacy_pinn_comparison_middle",
        "results/figures/comparison_t_99.png",
        "Legacy single-source PINN comparison, retained as backup.",
        "Appendix / Backup figures",
        4,
    ),
]


TABLE_CANDIDATES = [
    (
        "scenario_summary_multi_source",
        "results/data/synthetic_scenario_summary_multi_source.csv",
        "Final two-scenario multi-source synthetic summary.",
        "Results / Summary",
        1,
    ),
    (
        "model_comparison_multi_source",
        "results/data/model_comparison_scenario_summary_multi_source.csv",
        "Final comparison between PINN, traditional baselines, ablations, and seed repeats.",
        "Results / Model comparison",
        1,
    ),
    (
        "metrics_scenario_multi_source",
        "results/data/metrics_scenario_summary_multi_source.csv",
        "Per-snapshot PINN metrics for both final multi-source scenarios.",
        "Results / PINN accuracy",
        1,
    ),
    (
        "ablation_heterogeneous_multi_source",
        "results/data/ablation_summary_heterogeneous_multi_source.csv",
        "Ablation study for the heterogeneous multi-source case.",
        "Results / Ablation study",
        2,
    ),
    (
        "ablation_faulted_multi_source",
        "results/data/ablation_summary_faulted_multi_source.csv",
        "Ablation study for the faulted multi-source case.",
        "Results / Ablation study",
        2,
    ),
    (
        "multi_seed_heterogeneous_multi_source",
        "results/data/multi_seed_aggregate_heterogeneous_multi_source.csv",
        "Multi-seed robustness statistics for the heterogeneous multi-source case.",
        "Results / Robustness",
        2,
    ),
    (
        "multi_seed_faulted_multi_source",
        "results/data/multi_seed_aggregate_faulted_multi_source.csv",
        "Multi-seed robustness statistics for the faulted multi-source case.",
        "Results / Robustness",
        2,
    ),
    (
        "pde_residual_heterogeneous_multi_source",
        "results/data/pde_residual_summary_heterogeneous_multi_source.csv",
        "PDE residual diagnostics for the heterogeneous multi-source PINN.",
        "Results / Physical consistency",
        2,
    ),
    (
        "pde_residual_faulted_multi_source",
        "results/data/pde_residual_summary_faulted_multi_source.csv",
        "PDE residual diagnostics for the faulted multi-source PINN.",
        "Results / Physical consistency",
        2,
    ),
    (
        "fdm_metadata_heterogeneous_multi_source",
        "results/data/fdm_metadata_heterogeneous_multi_source.csv",
        "FDM stability, grid, source, and velocity metadata for the heterogeneous case.",
        "Methodology / Numerical setup",
        2,
    ),
    (
        "fdm_metadata_faulted_multi_source",
        "results/data/fdm_metadata_faulted_multi_source.csv",
        "FDM stability, grid, source, and velocity metadata for the faulted case.",
        "Methodology / Numerical setup",
        2,
    ),
    (
        "velocity_diagnostic_heterogeneous_multi_source",
        "results/data/velocity_inversion_diagnostic_heterogeneous_multi_source.csv",
        "Apparent-velocity diagnostic for the heterogeneous multi-source wavefield.",
        "Discussion / Inversion diagnostic",
        3,
    ),
    (
        "velocity_diagnostic_faulted_multi_source",
        "results/data/velocity_inversion_diagnostic_faulted_multi_source.csv",
        "Apparent-velocity diagnostic for the faulted multi-source wavefield.",
        "Discussion / Inversion diagnostic",
        3,
    ),
    (
        "real_validation",
        "results/data/real_physical_validation.csv",
        "Real-data validation-level diagnostics, if real mode was executed.",
        "Additional experiment / Real data",
        3,
    ),
    (
        "advanced_scope",
        "results/data/advanced_scope_matrix.csv",
        "Advanced-case scope matrix distinguishing implemented extensions from future work.",
        "Discussion / Future work",
        3,
    ),
    (
        "legacy_model_comparison",
        "results/data/model_comparison_summary.csv",
        "Legacy single-source comparison table, retained as backup.",
        "Appendix / Backup tables",
        4,
    ),
]


def _markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _write_manifest(kind, candidates, output_dir):
    rows = []
    for item_id, path, caption, section, priority in candidates:
        exists = Path(path).exists()
        rows.append(
            {
                "id": item_id,
                "path": path,
                "caption": caption,
                "thesis_section": section,
                "priority": priority,
                "exists": exists,
            }
        )

    csv_lines = ["id,path,caption,thesis_section,priority,exists"]
    for row in rows:
        csv_lines.append(
            ",".join(
                [
                    row["id"],
                    row["path"],
                    '"' + row["caption"].replace('"', '""') + '"',
                    '"' + row["thesis_section"].replace('"', '""') + '"',
                    str(row["priority"]),
                    str(row["exists"]),
                ]
            )
        )

    (output_dir / f"thesis_{kind}_manifest.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    markdown_rows = [
        [
            row["id"],
            row["path"],
            row["thesis_section"],
            row["priority"],
            "yes" if row["exists"] else "missing",
            row["caption"],
        ]
        for row in rows
    ]
    markdown = "# Thesis " + kind.title() + " Manifest\n\n"
    markdown += _markdown_table(
        ["id", "path", "section", "priority", "exists", "caption"],
        markdown_rows,
    )
    markdown += "\n"
    (output_dir / f"thesis_{kind}_manifest.md").write_text(markdown, encoding="utf-8")


def write_thesis_assets(output_dir="results/data"):
    """Write formula and manifest files for dissertation writing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "thesis_formulas.md").write_text(FORMULAS_MARKDOWN, encoding="utf-8")
    _write_manifest("figure", FIGURE_CANDIDATES, output_path)
    _write_manifest("table", TABLE_CANDIDATES, output_path)
