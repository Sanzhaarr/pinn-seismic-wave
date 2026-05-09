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

Homogeneous and layered FDM simulations are included as traditional baselines. This allows the defense to compare the neural result against simplified physical models and against the heterogeneous reference solution with anomalies.


## Synthetic experiment summary

     mode  num_snapshots  Mean_MSE  Mean_MAE  Mean_Relative_L2_Error  Mean_NRMSE  Mean_PSNR_dB  Mean_Correlation  Homogeneous_Mean_Relative_L2_Error  Homogeneous_Mean_NRMSE  Homogeneous_Mean_Correlation  Layered_Mean_Relative_L2_Error  Layered_Mean_NRMSE  Layered_Mean_Correlation  FDM_time_seconds  Homogeneous_FDM_time_seconds  Layered_FDM_time_seconds Neural_model_name Neural_display_name  PINN_training_time_seconds
synthetic              6  0.000004  0.001602                 0.37268    0.022486     33.143194          0.914146                             0.91795                0.139448                      0.376036                        0.424303            0.067746                  0.845473            0.0078                      0.006636                  0.007631      data_only_nn        Data-only NN                  777.779031

## Files to show during defense

- results/figures/velocity_model.png
- results/figures/fdm_snapshot_*.png
- results/figures/baseline_comparison_t_*.png
- results/figures/layered_baseline_comparison_t_*.png
- results/figures/layered_velocity_model.png
- results/figures/ricker_wavelet.png
- results/figures/comparison_t_*.png
- results/figures/loss_curve.png
- results/data/metrics.csv
- results/data/synthetic_summary.csv
- results/data/homogeneous_baseline_metrics.csv
- results/data/layered_baseline_metrics.csv
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