import numpy as np

from src.config import (
    NX,
    NT,
    NZ,
    SOURCE_AMPLITUDE,
    SOURCE_FREQUENCY,
    SOURCE_T0,
    SOURCE_X,
    SOURCE_Z,
    T_MAX,
    T_MIN,
    X_MAX,
    X_MIN,
    Z_MAX,
    Z_MIN,
)


def ricker_wavelet(t, f0=SOURCE_FREQUENCY, t0=SOURCE_T0, amplitude=SOURCE_AMPLITUDE):
    """
    Ricker wavelet used as a compact seismic source in time.

    Parameters
    ----------
    t : float or np.ndarray
        Time value(s).
    f0 : float
        Dominant frequency of the source.
    t0 : float
        Time shift controlling when the pulse is centered.
    amplitude : float
        Source amplitude.
    """
    pi2 = np.pi ** 2
    tau2 = (t - t0) ** 2
    return amplitude * (1.0 - 2.0 * pi2 * f0 ** 2 * tau2) * np.exp(-pi2 * f0 ** 2 * tau2)


def create_velocity_model(nx=NX, nz=NZ):
    """
    Create a simple heterogeneous velocity model.

    The model contains:
    1. a slower upper layer,
    2. a faster lower layer,
    3. a circular high-velocity anomaly.

    This structure is simple enough for a dissertation prototype but still clearly
    demonstrates propagation in a heterogeneous medium.
    """
    x = np.linspace(X_MIN, X_MAX, nx, dtype=np.float32)
    z = np.linspace(Z_MIN, Z_MAX, nz, dtype=np.float32)
    X, Z = np.meshgrid(x, z, indexing="ij")

    c = np.ones((nx, nz), dtype=np.float32)

    # Layered background medium.
    c[Z > 0.5] = 1.5

    # Smooth transition near the interface to avoid an unrealistically sharp jump.
    transition = 0.5 * (1.0 + np.tanh((Z - 0.5) / 0.025))
    c = (1.0 - transition) * 1.0 + transition * 1.5

    # Local high-velocity anomaly.
    anomaly = (X - 0.65) ** 2 + (Z - 0.35) ** 2 < 0.08 ** 2
    c[anomaly] = 2.0

    return x, z, c.astype(np.float32)


def build_absorbing_mask(nx, nz, damping_width=12, damping_strength=0.015):
    """
    Construct a smooth absorbing boundary mask.

    Values near the boundary are slightly below one and values in the interior are
    one. Applying this mask at every time step reduces artificial reflections.
    """
    mask = np.ones((nx, nz), dtype=np.float32)

    for i in range(nx):
        distance_to_x_boundary = min(i, nx - 1 - i)
        if distance_to_x_boundary < damping_width:
            normalized = (damping_width - distance_to_x_boundary) / damping_width
            mask[i, :] *= np.exp(-damping_strength * normalized ** 2)

    for j in range(nz):
        distance_to_z_boundary = min(j, nz - 1 - j)
        if distance_to_z_boundary < damping_width:
            normalized = (damping_width - distance_to_z_boundary) / damping_width
            mask[:, j] *= np.exp(-damping_strength * normalized ** 2)

    return mask


def solve_wave_equation_fdm():
    """
    Solve the 2D acoustic wave equation with a second-order finite-difference scheme.

    The equation is approximated as:
        u_tt = c(x,z)^2 (u_xx + u_zz) + s(x,z,t)

    The resulting wavefield is used as the reference solution for training and
    evaluating the PINN.
    """
    nx, nz, nt = NX, NZ, NT

    x, z, c = create_velocity_model(nx, nz)
    t = np.linspace(T_MIN, T_MAX, nt, dtype=np.float32)

    dx = (X_MAX - X_MIN) / (nx - 1)
    dz = (Z_MAX - Z_MIN) / (nz - 1)
    dt = (T_MAX - T_MIN) / (nt - 1)

    c_max = float(np.max(c))
    cfl = c_max * dt * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2)
    if cfl >= 1.0:
        raise ValueError(
            f"Unstable FDM configuration: CFL={cfl:.3f}. "
            "Increase NT or reduce NX/NZ/T_MAX to satisfy the CFL condition."
        )

    u = np.zeros((nt, nx, nz), dtype=np.float32)

    sx = int(np.argmin(np.abs(x - SOURCE_X)))
    sz = int(np.argmin(np.abs(z - SOURCE_Z)))

    damping_width = max(8, min(nx, nz) // 8)
    absorbing_mask = build_absorbing_mask(nx, nz, damping_width=damping_width)

    c2 = c ** 2
    dt2 = dt ** 2
    inv_dx2 = 1.0 / dx ** 2
    inv_dz2 = 1.0 / dz ** 2

    for n in range(1, nt - 1):
        laplacian = (
            (u[n, 2:, 1:-1] - 2.0 * u[n, 1:-1, 1:-1] + u[n, :-2, 1:-1]) * inv_dx2
            + (u[n, 1:-1, 2:] - 2.0 * u[n, 1:-1, 1:-1] + u[n, 1:-1, :-2]) * inv_dz2
        )

        u[n + 1, 1:-1, 1:-1] = (
            2.0 * u[n, 1:-1, 1:-1]
            - u[n - 1, 1:-1, 1:-1]
            + dt2 * c2[1:-1, 1:-1] * laplacian
        )

        # Point-source forcing. The dt^2 factor keeps the source consistent with
        # the second-order time discretization of u_tt.
        u[n + 1, sx, sz] += dt2 * ricker_wavelet(t[n])

        # Absorbing boundary to reduce artificial edge reflections.
        u[n + 1] *= absorbing_mask

    return x, z, t, c, u