import numpy as np
from src.config import *


def ricker_wavelet(t, f0=15.0, t0=0.12):
    """
    Ricker wavelet source function.
    """
    pi2 = np.pi ** 2
    return (1.0 - 2.0 * pi2 * f0 ** 2 * (t - t0) ** 2) * np.exp(
        -pi2 * f0 ** 2 * (t - t0) ** 2
    )


def create_velocity_model(nx=NX, nz=NZ):
    """
    Creates a heterogeneous velocity model with layers and a circular anomaly.
    """
    x = np.linspace(X_MIN, X_MAX, nx)
    z = np.linspace(Z_MIN, Z_MAX, nz)
    X, Z = np.meshgrid(x, z, indexing="ij")

    c = np.ones((nx, nz)) * 1.0

    # Layered medium
    c[Z > 0.5] = 1.5

    # Circular high-velocity anomaly
    anomaly = (X - 0.65) ** 2 + (Z - 0.35) ** 2 < 0.08 ** 2
    c[anomaly] = 2.0

    return x, z, c


def solve_wave_equation_fdm():
    """
    Solves 2D acoustic wave equation using finite difference method.
    Returns:
        x, z, t, c, u
    where:
        u has shape (nt, nx, nz)
    """
    nx, nz, nt = NX, NZ, NT

    x, z, c = create_velocity_model(nx, nz)
    t = np.linspace(T_MIN, T_MAX, nt)

    dx = (X_MAX - X_MIN) / (nx - 1)
    dz = (Z_MAX - Z_MIN) / (nz - 1)
    dt = (T_MAX - T_MIN) / (nt - 1)

    u = np.zeros((nt, nx, nz), dtype=np.float32)

    sx = np.argmin(np.abs(x - SOURCE_X))
    sz = np.argmin(np.abs(z - SOURCE_Z))

    for n in range(1, nt - 1):
        laplacian = (
            (u[n, 2:, 1:-1] - 2 * u[n, 1:-1, 1:-1] + u[n, :-2, 1:-1]) / dx ** 2
            + (u[n, 1:-1, 2:] - 2 * u[n, 1:-1, 1:-1] + u[n, 1:-1, :-2]) / dz ** 2
        )

        u[n + 1, 1:-1, 1:-1] = (
            2 * u[n, 1:-1, 1:-1]
            - u[n - 1, 1:-1, 1:-1]
            + dt ** 2 * c[1:-1, 1:-1] ** 2 * laplacian
        )

        # Source injection
        u[n + 1, sx, sz] += dt ** 2 * ricker_wavelet(t[n])

        # Simple absorbing damping near boundaries
        damping_width = 8
        damping_strength = 0.95

        u[n + 1, :damping_width, :] *= damping_strength
        u[n + 1, -damping_width:, :] *= damping_strength
        u[n + 1, :, :damping_width] *= damping_strength
        u[n + 1, :, -damping_width:] *= damping_strength

    return x, z, t, c, u