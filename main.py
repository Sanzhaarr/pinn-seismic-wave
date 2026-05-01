import os
import time
import pandas as pd

from src.fdm_solver import solve_wave_equation_fdm
from src.train_pinn import train_pinn, predict_snapshot
from src.metrics import mse_error, mae_error, relative_l2_error
from src.plots import (
    plot_velocity_model,
    plot_wave_snapshot,
    plot_comparison,
    plot_loss_curve
)


def main():
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)

    print("Step 1: Running finite difference simulation...")
    start_fdm = time.time()
    x, z, t, c, u_fdm = solve_wave_equation_fdm()
    fdm_time = time.time() - start_fdm

    print(f"FDM simulation completed in {fdm_time:.2f} seconds")

    print("Step 2: Saving velocity model and FDM snapshots...")
    plot_velocity_model(c)

    snapshot_indices = [60, 100, 140, 180]

    for idx in snapshot_indices:
        plot_wave_snapshot(
            u_fdm,
            idx,
            title=f"FDM Wavefield Snapshot at t={t[idx]:.3f}",
            path=f"results/figures/fdm_snapshot_{idx}.png"
        )

    print("Step 3: Training PINN...")
    start_pinn = time.time()
    model, loss_history = train_pinn(x, z, t, u_fdm)
    pinn_training_time = time.time() - start_pinn

    print(f"PINN training completed in {pinn_training_time:.2f} seconds")

    print("Step 4: Generating PINN predictions and metrics...")

    metrics_rows = []

    for idx in snapshot_indices:
        pinn_snapshot = predict_snapshot(model, time_value=t[idx])

        mse = mse_error(pinn_snapshot, u_fdm[idx])
        mae = mae_error(pinn_snapshot, u_fdm[idx])
        rel_l2 = relative_l2_error(pinn_snapshot, u_fdm[idx])

        metrics_rows.append({
            "time_index": idx,
            "time": t[idx],
            "MSE": mse,
            "MAE": mae,
            "Relative_L2_Error": rel_l2
        })

        plot_comparison(
            fdm=u_fdm,
            pinn=pinn_snapshot,
            time_index=idx,
            path=f"results/figures/comparison_t_{idx}.png"
        )

    print("Step 5: Saving loss curve and metrics...")
    plot_loss_curve(loss_history)

    df = pd.DataFrame(metrics_rows)
    df.to_csv("results/data/metrics.csv", index=False)

    print("\nFinal quantitative results:")
    print(df)

    print("\nProject completed.")
    print("Generated figures are saved in: results/figures")
    print("Metrics are saved in: results/data/metrics.csv")


if __name__ == "__main__":
    main()