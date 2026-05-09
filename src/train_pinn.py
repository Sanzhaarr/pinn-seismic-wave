import os

import numpy as np
import torch
from tqdm import tqdm

from src.config import *

from src.pinn_model import SeismicPINN


def set_global_seed(seed):
    """Set numpy and torch seeds for repeatable training comparisons."""
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def interpolate_velocity(x, z, model_type="heterogeneous"):
    """
    Analytical torch version of the synthetic heterogeneous velocity model.

    This must match src.fdm_solver.create_velocity_model as closely as possible,
    because the PDE residual uses this velocity model during PINN training.
    """
    if model_type == "homogeneous":
        return torch.ones_like(x) * 1.25

    transition = 0.5 * (1.0 + torch.tanh((z - 0.5) / 0.025))
    c = (1.0 - transition) * 1.0 + transition * 1.5

    if model_type == "layered":
        return c

    if model_type == "faulted":
        interface = 0.48 + 0.12 * (x - 0.5) + 0.04 * torch.sin(4.0 * torch.pi * x)
        interface = torch.where(x > 0.55, interface + 0.11, interface - 0.03)
        transition = 0.5 * (1.0 + torch.tanh((z - interface) / 0.022))
        c = (1.0 - transition) * 0.95 + transition * 1.55

        channel_center = 0.74 - 0.20 * x
        channel = (torch.abs(z - channel_center) < 0.045) & (x > 0.12) & (x < 0.88)
        c = torch.where(channel, torch.tensor(0.82, dtype=x.dtype, device=x.device), c)

        high_velocity_lens = ((x - 0.72) / 0.13) ** 2 + ((z - 0.28) / 0.08) ** 2 < 1.0
        c = torch.where(high_velocity_lens, torch.tensor(2.1, dtype=x.dtype, device=x.device), c)

        return c

    if model_type != "heterogeneous":
        raise ValueError(f"Unsupported velocity model_type={model_type!r}")

    anomaly = (x - 0.65) ** 2 + (z - 0.35) ** 2 < 0.08 ** 2
    c = torch.where(anomaly, torch.tensor(2.0, dtype=x.dtype, device=x.device), c)

    low_velocity_anomaly = (x - 0.35) ** 2 + (z - 0.70) ** 2 < 0.07 ** 2
    c = torch.where(low_velocity_anomaly, torch.tensor(0.8, dtype=x.dtype, device=x.device), c)

    return c


def pde_residual(model, xzt, velocity_model_type=None):
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
    if velocity_model_type is None:
        velocity_model_type = getattr(model, "velocity_model_type", "heterogeneous")
    c = interpolate_velocity(x, z, model_type=velocity_model_type)

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

    The sample mixes high-amplitude wavefront points, uniformly random points, and
    time-stratified points. The time-stratified part is important because global
    amplitude sampling can under-train early/late snapshots whose absolute energy
    is smaller but whose relative error is dissertation-visible.
    """
    flat_abs = np.abs(u_fdm).reshape(-1)
    all_indices = np.arange(flat_abs.size)

    n_high = int(0.45 * N_DATA)
    n_time_stratified = int(0.35 * N_DATA)
    n_uniform = N_DATA - n_high - n_time_stratified

    threshold = np.percentile(flat_abs, 70)
    high_amplitude_indices = np.where(flat_abs >= threshold)[0]

    if len(high_amplitude_indices) == 0:
        high_amplitude_indices = all_indices

    chosen_high = np.random.choice(high_amplitude_indices, size=n_high, replace=True)
    chosen_uniform = np.random.choice(all_indices, size=n_uniform, replace=True)

    time_indices = np.random.randint(0, len(t), size=n_time_stratified)
    x_indices = np.random.randint(0, len(x), size=n_time_stratified)
    z_indices = np.random.randint(0, len(z), size=n_time_stratified)
    chosen_time = np.ravel_multi_index((time_indices, x_indices, z_indices), u_fdm.shape)

    chosen = np.concatenate([chosen_high, chosen_time, chosen_uniform])
    np.random.shuffle(chosen)

    idx_t, idx_x, idx_z = np.unravel_index(chosen, u_fdm.shape)

    data_xzt = np.stack([x[idx_x], z[idx_z], t[idx_t]], axis=1)
    data_u = u_fdm[idx_t, idx_x, idx_z].reshape(-1, 1) / reference_scale

    data_xzt = torch.tensor(data_xzt, dtype=torch.float32, device=DEVICE)
    data_u = torch.tensor(data_u, dtype=torch.float32, device=DEVICE)

    return data_xzt, data_u


def _normalize_source_locations(source_locations=None):
    if source_locations is None:
        return [(SOURCE_X, SOURCE_Z, SOURCE_T0)]

    normalized = []
    for source in source_locations:
        if isinstance(source, dict):
            normalized.append(
                (
                    float(source.get("x", SOURCE_X)),
                    float(source.get("z", SOURCE_Z)),
                    float(source.get("t0", SOURCE_T0)),
                )
            )
        else:
            if len(source) == 2:
                sx, sz = source
                st0 = SOURCE_T0
            else:
                sx, sz, st0 = source[:3]
            normalized.append((float(sx), float(sz), float(st0)))
    return normalized


def sample_collocation_points(n_points=None, avoid_source=True, source_locations=None):
    n_points = N_COLLOCATION if n_points is None else int(n_points)

    x_c = torch.rand(n_points, 1, device=DEVICE)
    z_c = torch.rand(n_points, 1, device=DEVICE)
    t_c = torch.rand(n_points, 1, device=DEVICE) * T_MAX

    if avoid_source:
        radius2 = 0.035 ** 2
        for _ in range(4):
            near_source = torch.zeros_like(x_c, dtype=torch.bool)
            for sx, sz, st0 in _normalize_source_locations(source_locations):
                near_source = near_source | (
                    ((x_c - sx) ** 2 + (z_c - sz) ** 2 < radius2)
                    & (torch.abs(t_c - st0) < 0.08)
                )
            if not torch.any(near_source):
                break

            count = int(near_source.sum().item())
            x_c[near_source] = torch.rand(count, device=DEVICE)
            z_c[near_source] = torch.rand(count, device=DEVICE)
            t_c[near_source] = torch.rand(count, device=DEVICE) * T_MAX

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
    seed=None,
    velocity_model_type="heterogeneous",
    source_locations=None,
):
    active_seed = RANDOM_SEED if seed is None else int(seed)
    set_global_seed(active_seed)

    os.makedirs(DATA_DIR, exist_ok=True)

    num_epochs = EPOCHS if epochs is None else int(epochs)
    reference_scale = get_reference_scale(u_fdm)

    model = build_model(use_fourier=use_fourier)
    model.reference_scale = reference_scale
    model.experiment_name = experiment_name
    model.lambda_pde = float(lambda_pde)
    model.seed = active_seed
    model.velocity_model_type = velocity_model_type
    model.source_locations = _normalize_source_locations(source_locations)

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

    print(f"Experiment: {experiment_name}")
    print(f"Reference scale used for normalized training: {reference_scale:.6e}")
    preview_xzt, preview_u = prepare_training_data(x, z, t, u_fdm, reference_scale)
    print(f"Normalized training data range: min={preview_u.min().item():.6e}, max={preview_u.max().item():.6e}")
    print(f"PINN training device: {DEVICE}")
    if lambda_pde > 0:
        print(f"PDE residual weight: {lambda_pde:.2e} after warmup")
    else:
        print("PDE residual is disabled for optimization; residual diagnostics will still be evaluated after training.")

    loss_history = {
        "total": [],
        "pde": [],
        "data": [],
        "ic": [],
        "bc": [],
        "amplitude": [],
        "pde_weight": [],
        "lr": [],
    }

    pde_warmup_epochs = 0
    if lambda_pde > 0:
        pde_warmup_epochs = min(PDE_WARMUP_EPOCHS, max(0, num_epochs // 5))

    for epoch in tqdm(range(num_epochs), desc=f"Training PINN [{experiment_name}]"):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Resample supervised FDM points every epoch. This prevents the model from
        # memorizing one fixed subset and exploding between sampled points.
        data_xzt, data_u = prepare_training_data(x, z, t, u_fdm, reference_scale)

        pde_weight = float(lambda_pde) if epoch >= pde_warmup_epochs else 0.0
        if pde_weight > 0.0 and N_COLLOCATION > 0:
            xzt_c = sample_collocation_points(source_locations=model.source_locations)
            residual = pde_residual(model, xzt_c, velocity_model_type=velocity_model_type)
            loss_pde = torch.mean(residual ** 2)
        else:
            loss_pde = torch.zeros((), dtype=torch.float32, device=DEVICE)

        pred_data = model(data_xzt)
        loss_data = torch.mean((pred_data - data_u) ** 2)
        loss_amp = torch.mean(pred_data ** 2)

        xzt_i = sample_initial_points()
        pred_i = model(xzt_i)
        loss_ic = torch.mean(pred_i ** 2)

        xzt_b = sample_boundary_points()
        pred_b = model(xzt_b)
        loss_bc = torch.mean(pred_b ** 2)

        loss = (
            pde_weight * loss_pde
            + LAMBDA_DATA * loss_data
            + LAMBDA_IC * loss_ic
            + LAMBDA_BC * loss_bc
            + LAMBDA_AMPLITUDE * loss_amp
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
        loss_history["amplitude"].append(float(loss_amp.detach().cpu()))
        loss_history["pde_weight"].append(pde_weight)
        loss_history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if epoch % 250 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch} | "
                f"Total: {loss.item():.6e} | "
                f"PDE: {loss_pde.item():.6e} | "
                f"Data: {loss_data.item():.6e} | "
                f"IC: {loss_ic.item():.6e} | "
                f"BC: {loss_bc.item():.6e} | "
                f"Amp: {loss_amp.item():.6e} | "
                f"PDE_w: {pde_weight:.1e} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    # Final supervised sample for optional LBFGS
    data_xzt, data_u = prepare_training_data(x, z, t, u_fdm, reference_scale)
    if USE_LBFGS_REFINEMENT:
        _run_lbfgs_refinement(model, data_xzt, data_u, loss_history)

    checkpoint = {
        "state_dict": model.state_dict(),
        "reference_scale": reference_scale,
        "experiment_name": experiment_name,
        "use_fourier": use_fourier,
        "lambda_pde": lambda_pde,
        "seed": active_seed,
        "velocity_model_type": velocity_model_type,
        "source_locations": model.source_locations,
        "config_version": CONFIG_VERSION,
    }
    torch.save(checkpoint, f"{DATA_DIR}/pinn_model_{experiment_name}.pt")
    torch.save(checkpoint, f"{DATA_DIR}/pinn_model.pt")

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

        lambda_pde = getattr(model, "lambda_pde", LAMBDA_PDE)
        if lambda_pde > 0.0 and N_COLLOCATION > 0:
            xzt_c = sample_collocation_points(source_locations=getattr(model, "source_locations", None))
            residual = pde_residual(model, xzt_c)
            loss_pde = torch.mean(residual ** 2)
        else:
            loss_pde = torch.zeros((), dtype=torch.float32, device=DEVICE)

        pred_data = model(data_xzt)
        loss_data = torch.mean((pred_data - data_u) ** 2)
        loss_amp = torch.mean(pred_data ** 2)

        xzt_i = sample_initial_points()
        loss_ic = torch.mean(model(xzt_i) ** 2)

        xzt_b = sample_boundary_points()
        loss_bc = torch.mean(model(xzt_b) ** 2)

        loss = (
            lambda_pde * loss_pde
            + LAMBDA_DATA * loss_data
            + LAMBDA_IC * loss_ic
            + LAMBDA_BC * loss_bc
            + LAMBDA_AMPLITUDE * loss_amp
        )
        loss.backward()
        return loss

    final_loss = optimizer.step(closure)
    loss_history["total"].append(float(final_loss.detach().cpu()))


def evaluate_pde_residual(model, n_points=2048, batch_size=512):
    """Evaluate PDE residual statistics on unseen random collocation points."""
    model.eval()
    values = []

    remaining = int(n_points)
    while remaining > 0:
        current = min(int(batch_size), remaining)
        xzt_c = sample_collocation_points(
            n_points=current,
            source_locations=getattr(model, "source_locations", None),
        )
        residual = pde_residual(model, xzt_c)
        values.append(residual.detach().cpu().numpy().reshape(-1))
        remaining -= current

    values = np.concatenate(values) if values else np.array([], dtype=np.float32)
    if values.size == 0:
        return {
            "Residual_MAE": np.nan,
            "Residual_RMSE": np.nan,
            "Residual_MaxAbs": np.nan,
        }

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
