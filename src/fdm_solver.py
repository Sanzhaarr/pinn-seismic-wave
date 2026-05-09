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


def create_velocity_model(nx=NX, nz=NZ, model_type="heterogeneous"):
    """
    Create a velocity model for the FDM simulation.

    Parameters
    ----------
    nx, nz : int
        Number of grid points in the x and z directions.
    model_type : str
        "heterogeneous" for the main layered/anomaly model.
        "homogeneous" for a simple constant-velocity baseline.
        "layered" for a traditional layered-only FDM baseline without the anomaly.

    The homogeneous model is useful as a baseline in the dissertation because it
    shows what is lost when subsurface heterogeneity is ignored.
    """
    x = np.linspace(X_MIN, X_MAX, nx, dtype=np.float32)
    z = np.linspace(Z_MIN, Z_MAX, nz, dtype=np.float32)
    X, Z = np.meshgrid(x, z, indexing="ij")

    if model_type == "homogeneous":
        c = np.ones((nx, nz), dtype=np.float32) * 1.25
        return x, z, c

    if model_type not in {"heterogeneous", "layered"}:
        raise ValueError("model_type must be one of: 'heterogeneous', 'homogeneous', 'layered'")

    # Smooth layered background medium.
    transition = 0.5 * (1.0 + np.tanh((Z - 0.5) / 0.025))
    c = (1.0 - transition) * 1.0 + transition * 1.5

    if model_type == "layered":
        return x, z, c.astype(np.float32)

    # Local high-velocity anomaly. This makes the reference medium more complex
    # than the traditional homogeneous/layered baselines.
    high_velocity_anomaly = (X - 0.65) ** 2 + (Z - 0.35) ** 2 < 0.08 ** 2
    c[high_velocity_anomaly] = 2.0

    # Local low-velocity anomaly. Adding both high- and low-velocity inclusions
    # makes the heterogeneous reference more meaningful for a master-level project.
    low_velocity_anomaly = (X - 0.35) ** 2 + (Z - 0.70) ** 2 < 0.07 ** 2
    c[low_velocity_anomaly] = 0.8

    return x, z, c.astype(np.float32)


def build_absorbing_mask(nx, nz, damping_width=12, damping_strength=0.035):
    """
    Construct a smooth absorbing boundary mask.

    The mask is close to one in the interior and decreases smoothly toward the
    boundaries. Applying it at every time step reduces artificial reflections from
    the computational edges.
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

    return mask.astype(np.float32)


def find_source_indices(x, z):
    """Return integer grid indices closest to the configured source location."""
    sx = int(np.argmin(np.abs(x - SOURCE_X)))
    sz = int(np.argmin(np.abs(z - SOURCE_Z)))
    return sx, sz


def compute_fdm_stability_metadata(c, dx, dz, dt):
    """Return CFL and related numerical-stability metadata."""
    c_min = float(np.min(c))
    c_max = float(np.max(c))
    c_mean = float(np.mean(c))
    cfl = c_max * dt * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2)
    return {
        "c_min": c_min,
        "c_max": c_max,
        "c_mean": c_mean,
        "cfl": cfl,
    }


def solve_wave_equation_fdm(model_type="heterogeneous", return_metadata=False):
    """
    Solve the 2D acoustic wave equation with a second-order finite-difference scheme.

    The equation is approximated as:
        u_tt = c(x,z)^2 (u_xx + u_zz) + s(x,z,t)

    The heterogeneous wavefield is used as the main reference solution for training
    and evaluating the neural model. Homogeneous and layered wavefields are used as
    traditional FDM baselines.
    """
    nx, nz, nt = NX, NZ, NT

    x, z, c = create_velocity_model(nx, nz, model_type=model_type)
    t = np.linspace(T_MIN, T_MAX, nt, dtype=np.float32)

    dx = (X_MAX - X_MIN) / (nx - 1)
    dz = (Z_MAX - Z_MIN) / (nz - 1)
    dt = (T_MAX - T_MIN) / (nt - 1)

    stability = compute_fdm_stability_metadata(c, dx, dz, dt)
    cfl = stability["cfl"]
    if cfl >= 1.0:
        raise ValueError(
            f"Unstable FDM configuration: CFL={cfl:.3f}. "
            "Increase NT or reduce NX/NZ/T_MAX to satisfy the CFL condition."
        )

    u = np.zeros((nt, nx, nz), dtype=np.float32)

    sx, sz = find_source_indices(x, z)

    damping_width = max(10, min(nx, nz) // 7)
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

    if return_metadata:
        metadata = {
            "model_type": model_type,
            "dx": dx,
            "dz": dz,
            "dt": dt,
            "c_min": stability["c_min"],
            "c_max": stability["c_max"],
            "c_mean": stability["c_mean"],
            "cfl": cfl,
            "source_x_index": sx,
            "source_z_index": sz,
            "source_x": float(x[sx]),
            "source_z": float(z[sz]),
            "max_abs_amplitude": float(np.max(np.abs(u))),
            "damping_width": damping_width,
            "damping_min": float(np.min(absorbing_mask)),
            "damping_max": float(np.max(absorbing_mask)),
            "source_frequency": SOURCE_FREQUENCY,
            "source_amplitude": SOURCE_AMPLITUDE,
        }
        return x, z, t, c, u, metadata

    return x, z, t, c, u