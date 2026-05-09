# Defense Results Summary

Project topic:
Физически-информированные нейронные сети для моделирования распространения сейсмических волн в неоднородной среде

## Main proof used in the defense

The main proof is the controlled synthetic experiment. In this experiment, the heterogeneous velocity model, seismic source, initial conditions, boundary treatment, and finite-difference reference solution are known. This makes the comparison between FDM and PINN scientifically defensible.

The robustness extension includes a second faulted heterogeneous velocity model and optional multi-seed repeats. These additions help show that the result is not tied to one easy velocity model or one lucky random initialization.

The advanced-case extension implements the safe high-value additions: multiple seismic sources and an apparent-velocity diagnostic. Full 3D, elastic P/S-wave, and anisotropic PINNs are kept as future work because they require different governing equations, baselines, and validation data.

## Secondary real-data demonstration

The real seismic section is included as an applied demonstration. It shows that the neural representation can reconstruct a real seismic section, but it should be described as real-data reconstruction with physics-inspired smoothness regularization unless the exact field velocity model, acquisition geometry, and source wavelet are available. The file results/data/real_physical_validation.csv records this claim level explicitly.

## Recommended defense wording

The proposed PINN framework was evaluated in a controlled heterogeneous-medium experiment by comparing the neural approximation with a finite-difference reference solution of the acoustic wave equation. The real seismic section was then used as an additional reconstruction experiment to demonstrate practical applicability on field-like data.

## Added baseline validation

Homogeneous, layered, coarse-grid FDM, and sparse trilinear interpolation baselines are included as traditional baselines. This allows the defense to compare the neural result against simplified physical models, a cheaper numerical approximation, a non-neural reconstruction method, and the heterogeneous reference solution with anomalies.


## Synthetic experiment summary

     mode      scenario  source_case  num_snapshots  Mean_MSE  Mean_MAE  Mean_Relative_L2_Error  Mean_NRMSE  Mean_PSNR_dB  Mean_Correlation  Homogeneous_Mean_Relative_L2_Error  Homogeneous_Mean_NRMSE  Homogeneous_Mean_Correlation  Layered_Mean_Relative_L2_Error  Layered_Mean_NRMSE  Layered_Mean_Correlation  Coarse_Mean_Relative_L2_Error  Coarse_Mean_NRMSE  Coarse_Mean_Correlation  SparseInterp_Mean_Relative_L2_Error  SparseInterp_Mean_NRMSE  SparseInterp_Mean_Correlation  SparseInterp_sample_fraction  FDM_time_seconds  Homogeneous_FDM_time_seconds  Layered_FDM_time_seconds  Coarse_FDM_time_seconds  SparseInterp_time_seconds  VelocityDiagnostic_apparent_velocity                        Neural_model_name Neural_display_name  PINN_training_time_seconds
synthetic heterogeneous multi_source              6  0.000003  0.001269                0.153181    0.015922     36.142268          0.984259                            0.996834                0.157172                      0.344402                        0.502597            0.083453                  0.857611                       0.789517           0.119453                 0.650068                             0.558046                 0.072665                       0.839240                      0.018753          0.008286                      0.007213                  0.006518                 0.052198                   0.039191                              0.474763 weak_pde_pinn_heterogeneous_multi_source       Weak-PDE PINN                  266.248495
synthetic       faulted multi_source              6  0.000003  0.001296                0.162875    0.016525     35.856416          0.983521                            1.080483                0.152398                      0.334632                        1.013259            0.148298                  0.528407                       0.817733           0.112161                 0.598955                             0.567047                 0.069243                       0.832202                      0.018753          0.007296                      0.006928                  0.006507                 0.052129                   0.038257                              0.368871       weak_pde_pinn_faulted_multi_source       Weak-PDE PINN                  263.500010

## Files to show during defense

- results/figures/velocity_model.png
- results/figures/fdm_snapshot_*.png
- results/figures/baseline_comparison_t_*.png
- results/figures/layered_baseline_comparison_t_*.png
- results/figures/coarse_baseline_comparison_t_*.png
- results/figures/sparse_interpolation_comparison_t_*.png
- results/figures/layered_velocity_model.png
- results/figures/ricker_wavelet.png
- results/figures/comparison_t_*.png
- results/figures/loss_curve.png
- results/data/metrics.csv
- results/data/synthetic_summary.csv
- results/data/homogeneous_baseline_metrics.csv
- results/data/layered_baseline_metrics.csv
- results/data/coarse_baseline_metrics.csv
- results/data/sparse_interpolation_baseline_metrics.csv
- results/data/pde_residual_summary.csv
- results/data/fdm_metadata.csv
- results/data/model_comparison_summary.csv
- results/data/advanced_scope_matrix.csv
- results/data/advanced_scope_report.md
- results/data/velocity_inversion_diagnostic.csv
- results/figures/comparison_relative_l2_bar.png
- results/figures/comparison_correlation_bar.png
- results/figures/comparison_energy_ratio_bar.png
- results/data/ablation_summary.csv, if ablation mode was executed
- results/data/synthetic_scenario_summary.csv, if --scenario all was executed
- results/data/model_comparison_scenario_summary.csv, if --scenario all was executed
- results/data/*_multi_source.csv, if --source-case multi_source was executed
- results/data/multi_seed_summary.csv, if --multi-seed was executed
- results/data/multi_seed_aggregate.csv, if --multi-seed was executed
- results/data/real_metrics.csv, if real mode was executed
- results/data/real_physical_validation.csv, if real mode was executed
- results/data/real_physical_validation_report.md, if real mode was executed
- results/figures/real_validation_spectrum.png, if real mode was executed

## Important limitation

The real-data experiment alone is not sufficient as proof of physical wave propagation because the true velocity model, source function, and acquisition geometry are not fully constrained. The synthetic FDM-vs-PINN experiment should therefore remain the main scientific validation.