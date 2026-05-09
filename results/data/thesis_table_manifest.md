# Thesis Table Manifest

| id | path | section | priority | exists | caption |
| --- | --- | --- | --- | --- | --- |
| scenario_summary_multi_source | results/data/synthetic_scenario_summary_multi_source.csv | Results / Summary | 1 | yes | Final two-scenario multi-source synthetic summary. |
| model_comparison_multi_source | results/data/model_comparison_scenario_summary_multi_source.csv | Results / Model comparison | 1 | yes | Final comparison between PINN, traditional baselines, ablations, and seed repeats. |
| metrics_scenario_multi_source | results/data/metrics_scenario_summary_multi_source.csv | Results / PINN accuracy | 1 | yes | Per-snapshot PINN metrics for both final multi-source scenarios. |
| ablation_heterogeneous_multi_source | results/data/ablation_summary_heterogeneous_multi_source.csv | Results / Ablation study | 2 | yes | Ablation study for the heterogeneous multi-source case. |
| ablation_faulted_multi_source | results/data/ablation_summary_faulted_multi_source.csv | Results / Ablation study | 2 | yes | Ablation study for the faulted multi-source case. |
| multi_seed_heterogeneous_multi_source | results/data/multi_seed_aggregate_heterogeneous_multi_source.csv | Results / Robustness | 2 | yes | Multi-seed robustness statistics for the heterogeneous multi-source case. |
| multi_seed_faulted_multi_source | results/data/multi_seed_aggregate_faulted_multi_source.csv | Results / Robustness | 2 | yes | Multi-seed robustness statistics for the faulted multi-source case. |
| pde_residual_heterogeneous_multi_source | results/data/pde_residual_summary_heterogeneous_multi_source.csv | Results / Physical consistency | 2 | yes | PDE residual diagnostics for the heterogeneous multi-source PINN. |
| pde_residual_faulted_multi_source | results/data/pde_residual_summary_faulted_multi_source.csv | Results / Physical consistency | 2 | yes | PDE residual diagnostics for the faulted multi-source PINN. |
| fdm_metadata_heterogeneous_multi_source | results/data/fdm_metadata_heterogeneous_multi_source.csv | Methodology / Numerical setup | 2 | yes | FDM stability, grid, source, and velocity metadata for the heterogeneous case. |
| fdm_metadata_faulted_multi_source | results/data/fdm_metadata_faulted_multi_source.csv | Methodology / Numerical setup | 2 | yes | FDM stability, grid, source, and velocity metadata for the faulted case. |
| velocity_diagnostic_heterogeneous_multi_source | results/data/velocity_inversion_diagnostic_heterogeneous_multi_source.csv | Discussion / Inversion diagnostic | 3 | yes | Apparent-velocity diagnostic for the heterogeneous multi-source wavefield. |
| velocity_diagnostic_faulted_multi_source | results/data/velocity_inversion_diagnostic_faulted_multi_source.csv | Discussion / Inversion diagnostic | 3 | yes | Apparent-velocity diagnostic for the faulted multi-source wavefield. |
| real_validation | results/data/real_physical_validation.csv | Additional experiment / Real data | 3 | yes | Real-data validation-level diagnostics, if real mode was executed. |
| advanced_scope | results/data/advanced_scope_matrix.csv | Discussion / Future work | 3 | yes | Advanced-case scope matrix distinguishing implemented extensions from future work. |
| legacy_model_comparison | results/data/model_comparison_summary.csv | Appendix / Backup tables | 4 | yes | Legacy single-source comparison table, retained as backup. |
