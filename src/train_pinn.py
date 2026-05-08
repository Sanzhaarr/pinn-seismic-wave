import os

import numpy as np
import torch
from tqdm import tqdm

from src.config import *

from src.pinn_model import SeismicPINN


def interpolate_velocity(x, z):
    """
    Analytical torch version of the synthetic heterogeneous velocity model.

    This must match src.fdm_solver.create_velocity_model as closely as possible,
    because the PDE residual uses this velocity model during PINN training.
    """
    transition = 0.5 * (1.0 + torch.tanh((z - 0.5) / 0.025))
    c = (1.0 - transition) * 1.0 + transition * 1.5

    anomaly = (x - 0.65) ** 2 + (z - 0.35) ** 2 < 0.08 ** 2
    c = torch.where(anomaly, torch.tensor(2.0, dtype=x.dtype, device=x.device), c)

    return c




def pde_residual(model, xzt):
    """
    Compute the homogeneous acoustic wave-equation residual:

        u_tt - c(x,z)^2 (u_xx + u_zz) = 0

    In this project, the source-generated wavefield is learned primarily from the
    FDM data term. The PDE residual is kept as a weak regularizer so that the neural
    approximation remains wave-equation-like without allowing the source term to
    dominate optimization.
    """
    xzt = xzt.clone().detach().requires_grad_(True)
    u = model(xzt)

    grads = torch.autograd.grad(
        u,
        xzt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_x = grads[:, 0:1]
    u_z = grads[:, 1:2]
    u_t = grads[:, 2:3]

    u_xx = torch.autograd.grad(
        u_x,
        xzt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0][:, 0:1]

    u_zz = torch.autograd.grad(
        u_z,
        xzt,
        grad_outputs=torch.ones_like(u_z),
        create_graph=True,
    )[0][:, 1:2]

    u_tt = torch.autograd.grad(
        u_t,
        xzt,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True,
    )[0][:, 2:3]

    x = xzt[:, 0:1]
    z = xzt[:, 1:2]
    c = interpolate_velocity(x, z)

    return u_tt - c ** 2 * (u_xx + u_zz)

def get_reference_scale(u_fdm):
    """Return a stable amplitude scale for normalizing the FDM wavefield."""
    scale = float(np.max(np.abs(u_fdm)))
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def prepare_training_data(x, z, t, u_fdm, reference_scale):
    """
    Sample supervised data points from the FDM solution.

    Half of the points are sampled from high-amplitude regions and half are sampled
    uniformly. This keeps important wavefronts while still covering quiet regions.
    """
    flat_abs = np.abs(u_fdm).reshape(-1)
    all_indices = np.arange(flat_abs.size)

    n_high = N_DATA // 2
    n_uniform = N_DATA - n_high

    threshold = np.percentile(flat_abs, 70)
    high_amplitude_indices = np.where(flat_abs >= threshold)[0]

    if len(high_amplitude_indices) == 0:
        high_amplitude_indices = all_indices

    chosen_high = np.random.choice(high_amplitude_indices, size=n_high, replace=True)
    chosen_uniform = np.random.choice(all_indices, size=n_uniform, replace=True)
    chosen = np.concatenate([chosen_high, chosen_uniform])
    np.random.shuffle(chosen)

    idx_t, idx_x, idx_z = np.unravel_index(chosen, u_fdm.shape)

    data_xzt = np.stack([x[idx_x], z[idx_z], t[idx_t]], axis=1)
    data_u = u_fdm[idx_t, idx_x, idx_z].reshape(-1, 1) / reference_scale

    data_xzt = torch.tensor(data_xzt, dtype=torch.float32, device=DEVICE)
    data_u = torch.tensor(data_u, dtype=torch.float32, device=DEVICE)

    return data_xzt, data_u


def sample_collocation_points():
    x_c = torch.rand(N_COLLOCATION, 1, device=DEVICE)
    z_c = torch.rand(N_COLLOCATION, 1, device=DEVICE)
    t_c = torch.rand(N_COLLOCATION, 1, device=DEVICE) * T_MAX
    return torch.cat([x_c, z_c, t_c], dim=1)


def sample_initial_points():
    x_i = torch.rand(N_INITIAL, 1, device=DEVICE)
    z_i = torch.rand(N_INITIAL, 1, device=DEVICE)
    t_i = torch.zeros(N_INITIAL, 1, device=DEVICE)
    return torch.cat([x_i, z_i, t_i], dim=1)


def sample_boundary_points():
    n_b = max(1, N_BOUNDARY // 4)
    tb = torch.rand(n_b, 1, device=DEVICE) * T_MAX

    xb0 = torch.zeros(n_b, 1, device=DEVICE)
    xb1 = torch.ones(n_b, 1, device=DEVICE)
    zb = torch.rand(n_b, 1, device=DEVICE)

    zb0 = torch.zeros(n_b, 1, device=DEVICE)
    zb1 = torch.ones(n_b, 1, device=DEVICE)
    xb = torch.rand(n_b, 1, device=DEVICE)

    b1 = torch.cat([xb0, zb, tb], dim=1)
    b2 = torch.cat([xb1, zb, tb], dim=1)
    b3 = torch.cat([xb, zb0, tb], dim=1)
    b4 = torch.cat([xb, zb1, tb], dim=1)

    return torch.cat([b1, b2, b3, b4], dim=0)


def build_model(use_fourier=USE_FOURIER_FEATURES):
    return SeismicPINN(
        use_fourier=use_fourier,
        mapping_size=FOURIER_MAPPING_SIZE,
        fourier_scale=FOURIER_SCALE,
        hidden_dim=HIDDEN_DIM,
        depth=NETWORK_DEPTH,
        activation=ACTIVATION,
    ).to(DEVICE)


def train_pinn(
    x,
    z,
    t,
    u_fdm,
    experiment_name="full",
    use_fourier=USE_FOURIER_FEATURES,
    lambda_pde=LAMBDA_PDE,
    epochs=None,
):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(DATA_DIR, exist_ok=True)

    num_epochs = EPOCHS if epochs is None else int(epochs)
    reference_scale = get_reference_scale(u_fdm)

    model = build_model(use_fourier=use_fourier)
    model.reference_scale = reference_scale
    model.experiment_name = experiment_name

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=250,
    )

    data_xzt, data_u = prepare_training_data(x, z, t, u_fdm, reference_scale)
    print(f"Experiment: {experiment_name}")
    print(f"Reference scale used for normalized training: {reference_scale:.6e}")
    print(f"Normalized training data range: min={data_u.min().item():.6e}, max={data_u.max().item():.6e}")
    print(f"PINN training device: {DEVICE}")

    loss_history = {
        "total": [],
        "pde": [],
        "data": [],
        "ic": [],
        "bc": [],
        "lr": [],
    }

    for epoch in tqdm(range(num_epochs), desc=f"Training PINN [{experiment_name}]"):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        xzt_c = sample_collocation_points()
        residual = pde_residual(model, xzt_c)
        loss_pde = torch.mean(residual ** 2)

        pred_data = model(data_xzt)
        loss_data = torch.mean((pred_data - data_u) ** 2)

        xzt_i = sample_initial_points()
        pred_i = model(xzt_i)
        loss_ic = torch.mean(pred_i ** 2)

        xzt_b = sample_boundary_points()
        pred_b = model(xzt_b)
        loss_bc = torch.mean(pred_b ** 2)

        loss = (
            lambda_pde * loss_pde
            + LAMBDA_DATA * loss_data
            + LAMBDA_IC * loss_ic
            + LAMBDA_BC * loss_bc
        )

        loss.backward()

        if GRADIENT_CLIP_NORM is not None and GRADIENT_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)

        optimizer.step()
        scheduler.step(loss.detach())

        loss_history["total"].append(float(loss.detach().cpu()))
        loss_history["pde"].append(float(loss_pde.detach().cpu()))
        loss_history["data"].append(float(loss_data.detach().cpu()))
        loss_history["ic"].append(float(loss_ic.detach().cpu()))
        loss_history["bc"].append(float(loss_bc.detach().cpu()))
        loss_history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if epoch % 250 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch} | "
                f"Total: {loss.item():.6e} | "
                f"PDE: {loss_pde.item():.6e} | "
                f"Data: {loss_data.item():.6e} | "
                f"IC: {loss_ic.item():.6e} | "
                f"BC: {loss_bc.item():.6e} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    if USE_LBFGS_REFINEMENT:
        _run_lbfgs_refinement(model, data_xzt, data_u, loss_history)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "reference_scale": reference_scale,
            "experiment_name": experiment_name,
            "use_fourier": use_fourier,
            "lambda_pde": lambda_pde,
            "config_version": CONFIG_VERSION,
        },
        f"{DATA_DIR}/pinn_model_{experiment_name}.pt",
    )

    return model, loss_history


def _run_lbfgs_refinement(model, data_xzt, data_u, loss_history):
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=LBFGS_MAX_ITER,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)

        xzt_c = sample_collocation_points()
        residual = pde_residual(model, xzt_c)
        loss_pde = torch.mean(residual ** 2)

        pred_data = model(data_xzt)
        loss_data = torch.mean((pred_data - data_u) ** 2)

        xzt_i = sample_initial_points()
        loss_ic = torch.mean(model(xzt_i) ** 2)

        xzt_b = sample_boundary_points()
        loss_bc = torch.mean(model(xzt_b) ** 2)

        loss = (
            getattr(model, "lambda_pde", LAMBDA_PDE) * loss_pde
            + LAMBDA_DATA * loss_data
            + LAMBDA_IC * loss_ic
            + LAMBDA_BC * loss_bc
        )
        loss.backward()
        return loss

    final_loss = optimizer.step(closure)
    loss_history["total"].append(float(final_loss.detach().cpu()))


def evaluate_pde_residual(model, n_points=5000):
    """Evaluate PDE residual statistics on unseen random collocation points."""
    model.eval()
    xzt_c = sample_collocation_points()
    if xzt_c.shape[0] > n_points:
        xzt_c = xzt_c[:n_points]
    residual = pde_residual(model, xzt_c)
    values = residual.detach().cpu().numpy().reshape(-1)
    return {
        "Residual_MAE": float(np.mean(np.abs(values))),
        "Residual_RMSE": float(np.sqrt(np.mean(values ** 2))),
        "Residual_MaxAbs": float(np.max(np.abs(values))),
    }


def predict_snapshot(model, time_value, nx=NX, nz=NZ):
    x = np.linspace(X_MIN, X_MAX, nx, dtype=np.float32)
    z = np.linspace(Z_MIN, Z_MAX, nz, dtype=np.float32)

    X, Z = np.meshgrid(x, z, indexing="ij")
    T = np.ones_like(X, dtype=np.float32) * time_value

    xzt = np.stack([X.flatten(), Z.flatten(), T.flatten()], axis=1)
    xzt_tensor = torch.tensor(xzt, dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(xzt_tensor).cpu().numpy().reshape(nx, nz)

    reference_scale = getattr(model, "reference_scale", 1.0)
    return pred * reference_scale