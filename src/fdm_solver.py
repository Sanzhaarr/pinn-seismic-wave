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


def normalize_sources(sources=None):
    """Return source dictionaries with all source parameters filled in."""
    if not sources:
        sources = [
            {
                "x": SOURCE_X,
                "z": SOURCE_Z,
                "frequency": SOURCE_FREQUENCY,
                "t0": SOURCE_T0,
                "amplitude": SOURCE_AMPLITUDE,
            }
        ]

    normalized = []
    for index, source in enumerate(sources):
        normalized.append(
            {
                "id": source.get("id", f"source_{index}"),
                "x": float(source.get("x", SOURCE_X)),
                "z": float(source.get("z", SOURCE_Z)),
                "frequency": float(source.get("frequency", SOURCE_FREQUENCY)),
                "t0": float(source.get("t0", SOURCE_T0)),
                "amplitude": float(source.get("amplitude", SOURCE_AMPLITUDE)),
            }
        )
    return normalized


def create_velocity_model(nx=NX, nz=NZ, model_type="heterogeneous"):
    """
    Create a velocity model for the FDM simulation.

    Parameters
    ----------
    nx, nz : int
        Number of grid points in the x and z directions.
    model_type : str
        "heterogeneous" for the main layered/anomaly model.
        "faulted" for a second, harder heterogeneous validation model.
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

    if model_type not in {"heterogeneous", "faulted", "layered"}:
        raise ValueError("model_type must be one of: 'heterogeneous', 'faulted', 'homogeneous', 'layered'")

    # Smooth layered background medium.
    transition = 0.5 * (1.0 + np.tanh((Z - 0.5) / 0.025))
    c = (1.0 - transition) * 1.0 + transition * 1.5

    if model_type == "layered":
        return x, z, c.astype(np.float32)

    if model_type == "faulted":
        # A second heterogeneous scenario for robustness testing. The interface
        # is laterally shifted by a simple fault, with a low-velocity channel and
        # a compact high-velocity lens. This differs materially from the primary
        # circular-anomaly case without requiring a new solver.
        interface = 0.48 + 0.12 * (X - 0.5) + 0.04 * np.sin(4.0 * np.pi * X)
        interface = np.where(X > 0.55, interface + 0.11, interface - 0.03)
        transition = 0.5 * (1.0 + np.tanh((Z - interface) / 0.022))
        c = (1.0 - transition) * 0.95 + transition * 1.55

        channel_center = 0.74 - 0.20 * X
        channel = (np.abs(Z - channel_center) < 0.045) & (X > 0.12) & (X < 0.88)
        c[channel] = 0.82

        high_velocity_lens = ((X - 0.72) / 0.13) ** 2 + ((Z - 0.28) / 0.08) ** 2 < 1.0
        c[high_velocity_lens] = 2.1

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


def find_source_indices(x, z, source_x=SOURCE_X, source_z=SOURCE_Z):
    """Return integer grid indices closest to the configured source location."""
    sx = int(np.argmin(np.abs(x - source_x)))
    sz = int(np.argmin(np.abs(z - source_z)))
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


def solve_wave_equation_fdm(
    model_type="heterogeneous",
    return_metadata=False,
    nx=None,
    nz=None,
    nt=None,
    sources=None,
):
    """
    Solve the 2D acoustic wave equation with a second-order finite-difference scheme.

    The equation is approximated as:
        u_tt = c(x,z)^2 (u_xx + u_zz) + s(x,z,t)

    The heterogeneous wavefield is used as the main reference solution for training
    and evaluating the neural model. Homogeneous and layered wavefields are used as
    traditional FDM baselines.
    """
    nx = NX if nx is None else int(nx)
    nz = NZ if nz is None else int(nz)
    nt = NT if nt is None else int(nt)

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

    active_sources = normalize_sources(sources)
    source_grid = []
    for source in active_sources:
        sx, sz = find_source_indices(x, z, source_x=source["x"], source_z=source["z"])
        source_grid.append((sx, sz, source))

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

        # Point-source forcing. Dividing by the cell area makes the discrete
        # delta source comparable across grid resolutions.
        for sx, sz, source in source_grid:
            u[n + 1, sx, sz] += (
                dt2
                * ricker_wavelet(
                    t[n],
                    f0=source["frequency"],
                    t0=source["t0"],
                    amplitude=source["amplitude"],
                )
                / (dx * dz)
            )

        # Absorbing boundary to reduce artificial edge reflections.
        u[n + 1] *= absorbing_mask

    if return_metadata:
        first_sx, first_sz, first_source = source_grid[0]
        source_summary = "; ".join(
            (
                f"{source['id']}@"
                f"({float(x[sx]):.4f},{float(z[sz]):.4f}),"
                f"f={source['frequency']:.3f},"
                f"t0={source['t0']:.3f},"
                f"amp={source['amplitude']:.3f}"
            )
            for sx, sz, source in source_grid
        )
        metadata = {
            "model_type": model_type,
            "nx": nx,
            "nz": nz,
            "nt": nt,
            "dx": dx,
            "dz": dz,
            "dt": dt,
            "c_min": stability["c_min"],
            "c_max": stability["c_max"],
            "c_mean": stability["c_mean"],
            "cfl": cfl,
            "num_sources": len(source_grid),
            "source_x_index": first_sx,
            "source_z_index": first_sz,
            "source_x": float(x[first_sx]),
            "source_z": float(z[first_sz]),
            "source_summary": source_summary,
            "max_abs_amplitude": float(np.max(np.abs(u))),
            "damping_width": damping_width,
            "damping_min": float(np.min(absorbing_mask)),
            "damping_max": float(np.max(absorbing_mask)),
            "source_frequency": first_source["frequency"],
            "source_amplitude": first_source["amplitude"],
            "source_discretization": "point_delta_scaled_by_cell_area",
        }
        return x, z, t, c, u, metadata

    return x, z, t, c, u
