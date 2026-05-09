# Defense Results Summary

Project topic:
Физически-информированные нейронные сети для моделирования распространения сейсмических волн в неоднородной среде

## Main proof used in the defense

The main proof is the controlled synthetic experiment. In this experiment, the heterogeneous velocity model, seismic source, initial conditions, boundary treatment, and finite-difference reference solution are known. This makes the comparison between FDM and PINN scientifically defensible.

## Secondary real-data demonstration

The real seismic section is included as an applied demonstration. It shows that the neural representation can reconstruct a real seismic section, but it should be described as real-data reconstruction with physics-inspired smoothness regularization unless the exact field velocity model, acquisition geometry, and source wavelet are available.

## Recommended defense wording

The proposed PINN framework was evaluated in a controlled heterogeneous-medium experiment by comparing the neural approximation with a finite-difference reference solution of the acoustic wave equation. The real seismic section was then used as an additional reconstruction experiment to demonstrate practical applicability on field-like data.

## Added baseline validation

Homogeneous, layered, coarse-grid FDM, and sparse trilinear interpolation baselines are included as traditional baselines. This allows the defense to compare the neural result against simplified physical models, a cheaper numerical approximation, a non-neural reconstruction method, and the heterogeneous reference solution with anomalies.


## Synthetic experiment summary

     mode  num_snapshots  Mean_MSE  Mean_MAE  Mean_Relative_L2_Error  Mean_NRMSE  Mean_PSNR_dB  Mean_Correlation  Homogeneous_Mean_Relative_L2_Error  Homogeneous_Mean_NRMSE  Homogeneous_Mean_Correlation  Layered_Mean_Relative_L2_Error  Layered_Mean_NRMSE  Layered_Mean_Correlation  Coarse_Mean_Relative_L2_Error  Coarse_Mean_NRMSE  Coarse_Mean_Correlation  SparseInterp_Mean_Relative_L2_Error  SparseInterp_Mean_NRMSE  SparseInterp_Mean_Correlation  SparseInterp_sample_fraction  FDM_time_seconds  Homogeneous_FDM_time_seconds  Layered_FDM_time_seconds  Coarse_FDM_time_seconds  SparseInterp_time_seconds Neural_model_name Neural_display_name  PINN_training_time_seconds
synthetic              6  0.000002  0.001067                0.242569    0.015778     36.297782          0.952448                             0.91795                0.139448                      0.376036                        0.424303            0.067746                  0.845473                       0.756462           0.102392                 0.695515                             0.592434                 0.063264                       0.804804                      0.018753          0.006975                      0.006101                  0.006668                 0.075148                   0.039167     weak_pde_pinn       Weak-PDE PINN                  361.447305

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
- results/figures/comparison_relative_l2_bar.png
- results/figures/comparison_correlation_bar.png
- results/figures/comparison_energy_ratio_bar.png
- results/data/ablation_summary.csv, if ablation mode was executed
- results/data/real_metrics.csv, if real mode was executed

## Important limitation

The real-data experiment alone is not sufficient as proof of physical wave propagation because the true velocity model, source function, and acquisition geometry are not fully constrained. The synthetic FDM-vs-PINN experiment should therefore remain the main scientific validation.