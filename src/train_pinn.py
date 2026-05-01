import torch
import numpy as np
from tqdm import tqdm

from src.config import *
from src.pinn_model import SeismicPINN


def interpolate_velocity(x, z):
    """
    Analytical version of the same heterogeneous velocity model.
    This avoids complicated interpolation inside the PINN.
    """
    c = torch.ones_like(x)

    # Layer
    c = torch.where(z > 0.5, torch.tensor(1.5, device=x.device), c)

    # Circular anomaly
    anomaly = (x - 0.65) ** 2 + (z - 0.35) ** 2 < 0.08 ** 2
    c = torch.where(anomaly, torch.tensor(2.0, device=x.device), c)

    return c


def pde_residual(model, xzt):
    """
    Computes residual:
    u_tt - c(x,z)^2 * (u_xx + u_zz) = 0
    """
    xzt.requires_grad_(True)

    u = model(xzt)

    grads = torch.autograd.grad(
        u,
        xzt,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grads[:, 0:1]
    u_z = grads[:, 1:2]
    u_t = grads[:, 2:3]

    u_xx = torch.autograd.grad(
        u_x,
        xzt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0:1]

    u_zz = torch.autograd.grad(
        u_z,
        xzt,
        grad_outputs=torch.ones_like(u_z),
        create_graph=True
    )[0][:, 1:2]

    u_tt = torch.autograd.grad(
        u_t,
        xzt,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True
    )[0][:, 2:3]

    x = xzt[:, 0:1]
    z = xzt[:, 1:2]
    c = interpolate_velocity(x, z)

    residual = u_tt - c ** 2 * (u_xx + u_zz)

    return residual


def prepare_training_data(x, z, t, u_fdm):
    """
    Samples data points from FDM solution.
    """
    nx = len(x)
    nz = len(z)
    nt = len(t)

    # Data points from reference FDM
    idx_x = np.random.randint(0, nx, N_DATA)
    idx_z = np.random.randint(0, nz, N_DATA)
    idx_t = np.random.randint(0, nt, N_DATA)

    data_xzt = np.stack([
        x[idx_x],
        z[idx_z],
        t[idx_t]
    ], axis=1)

    data_u = u_fdm[idx_t, idx_x, idx_z].reshape(-1, 1)

    data_xzt = torch.tensor(data_xzt, dtype=torch.float32).to(DEVICE)
    data_u = torch.tensor(data_u, dtype=torch.float32).to(DEVICE)

    return data_xzt, data_u


def train_pinn(x, z, t, u_fdm):
    model = SeismicPINN(use_fourier=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    data_xzt, data_u = prepare_training_data(x, z, t, u_fdm)

    loss_history = []

    for epoch in tqdm(range(EPOCHS), desc="Training PINN"):
        optimizer.zero_grad()

        # Collocation points
        x_c = torch.rand(N_COLLOCATION, 1).to(DEVICE)
        z_c = torch.rand(N_COLLOCATION, 1).to(DEVICE)
        t_c = torch.rand(N_COLLOCATION, 1).to(DEVICE) * T_MAX
        xzt_c = torch.cat([x_c, z_c, t_c], dim=1)

        residual = pde_residual(model, xzt_c)
        loss_pde = torch.mean(residual ** 2)

        # Data loss
        pred_data = model(data_xzt)
        loss_data = torch.mean((pred_data - data_u) ** 2)

        # Initial condition: u(x,z,0) = 0
        x_i = torch.rand(N_INITIAL, 1).to(DEVICE)
        z_i = torch.rand(N_INITIAL, 1).to(DEVICE)
        t_i = torch.zeros(N_INITIAL, 1).to(DEVICE)
        xzt_i = torch.cat([x_i, z_i, t_i], dim=1)
        pred_i = model(xzt_i)
        loss_ic = torch.mean(pred_i ** 2)

        # Boundary condition: u = 0 at boundaries
        n_b = N_BOUNDARY // 4
        tb = torch.rand(n_b, 1).to(DEVICE) * T_MAX

        xb0 = torch.zeros(n_b, 1).to(DEVICE)
        xb1 = torch.ones(n_b, 1).to(DEVICE)
        zb = torch.rand(n_b, 1).to(DEVICE)

        zb0 = torch.zeros(n_b, 1).to(DEVICE)
        zb1 = torch.ones(n_b, 1).to(DEVICE)
        xb = torch.rand(n_b, 1).to(DEVICE)

        b1 = torch.cat([xb0, zb, tb], dim=1)
        b2 = torch.cat([xb1, zb, tb], dim=1)
        b3 = torch.cat([xb, zb0, tb], dim=1)
        b4 = torch.cat([xb, zb1, tb], dim=1)

        pred_b = torch.cat([
            model(b1),
            model(b2),
            model(b3),
            model(b4)
        ], dim=0)

        loss_bc = torch.mean(pred_b ** 2)

        loss = (
            LAMBDA_PDE * loss_pde
            + LAMBDA_DATA * loss_data
            + LAMBDA_IC * loss_ic
            + LAMBDA_BC * loss_bc
        )

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch} | "
                f"Total: {loss.item():.6e} | "
                f"PDE: {loss_pde.item():.6e} | "
                f"Data: {loss_data.item():.6e}"
            )

    torch.save(model.state_dict(), "results/data/pinn_model.pt")

    return model, loss_history


def predict_snapshot(model, time_value, nx=NX, nz=NZ):
    x = np.linspace(X_MIN, X_MAX, nx)
    z = np.linspace(Z_MIN, Z_MAX, nz)

    X, Z = np.meshgrid(x, z, indexing="ij")
    T = np.ones_like(X) * time_value

    xzt = np.stack([X.flatten(), Z.flatten(), T.flatten()], axis=1)
    xzt_tensor = torch.tensor(xzt, dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(xzt_tensor).cpu().numpy().reshape(nx, nz)

    return pred