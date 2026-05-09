import os

import torch

# -----------------------------------------------------------------------------
# Global reproducibility and device settings
# -----------------------------------------------------------------------------
def _select_device():
    """Pick the fastest available PyTorch device, with an env override for debugging."""
    forced_device = os.getenv("SEISMIC_DEVICE", "").strip().lower()
    if forced_device:
        return torch.device(forced_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = _select_device()
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Computational domain
# -----------------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 0.6

# -----------------------------------------------------------------------------
# Finite-difference reference solver
# -----------------------------------------------------------------------------
# Defense-safe values for a laptop run.
NX = 70
NZ = 70
NT = 220

# Coarse traditional FDM baseline. This solves the same heterogeneous physics on
# a cheaper grid, then interpolates back to the reference grid for comparison.
COARSE_NX = 36
COARSE_NZ = 36
COARSE_NT = NT

# Traditional non-neural reconstruction baseline. A sparse subset of the full
# FDM wavefield is interpolated back to the full grid and compared with the
# reference. This tests whether the PINN is doing more than simple interpolation.
SPARSE_BASELINE_TIME_STRIDE = 4
SPARSE_BASELINE_SPACE_STRIDE = 4

# Robustness scenarios for dissertation experiments. The first scenario keeps
# backward-compatible filenames; additional scenarios write suffixed artifacts.
SYNTHETIC_SCENARIOS = ["heterogeneous", "faulted"]

# -----------------------------------------------------------------------------
# Seismic source settings
# -----------------------------------------------------------------------------
SOURCE_X = 0.5
SOURCE_Z = 0.5
SOURCE_FREQUENCY = 8.0
SOURCE_T0 = 0.15
SOURCE_AMPLITUDE = 1.0

# Optional multi-shot source case. This is a useful extension because it tests
# whether the same PINN machinery can learn interacting wavefronts without
# changing the governing equation.
MULTI_SOURCE_DEFINITIONS = [
    {"id": "left_shot", "x": 0.28, "z": 0.36, "frequency": 8.0, "t0": 0.12, "amplitude": 0.75},
    {"id": "center_shot", "x": 0.50, "z": 0.50, "frequency": 8.0, "t0": 0.15, "amplitude": 0.80},
    {"id": "right_shot", "x": 0.72, "z": 0.34, "frequency": 10.0, "t0": 0.18, "amplitude": 0.65},
]

# -----------------------------------------------------------------------------
# PINN architecture defaults
# -----------------------------------------------------------------------------
USE_FOURIER_FEATURES = True
FOURIER_MAPPING_SIZE = 128
# Lower Fourier scale reduces high-frequency over-amplification between sampled points.
FOURIER_SCALE = 3.0
HIDDEN_DIM = 128
NETWORK_DEPTH = 6
ACTIVATION = "tanh"

# -----------------------------------------------------------------------------
# PINN training settings
# -----------------------------------------------------------------------------
N_COLLOCATION = 1024
N_DATA = 32000
N_INITIAL = 2000
N_BOUNDARY = 2000

EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0

USE_LBFGS_REFINEMENT = False
LBFGS_MAX_ITER = 300

# -----------------------------------------------------------------------------
# Loss weights
# -----------------------------------------------------------------------------
# The PDE residual is numerically large because the network output is
# normalized and the model uses Fourier features. A very small nonzero weight
# keeps the experiment genuinely physics-informed without destroying the data fit.
LAMBDA_PDE = 1e-8
LAMBDA_DATA = 100.0
LAMBDA_IC = 0.5
LAMBDA_BC = 0.5
LAMBDA_AMPLITUDE = 0.01
PDE_WARMUP_EPOCHS = 200

# Real sections usually lack a known velocity/source model. Use smoothness as a
# reconstruction regularizer instead of pretending the full wave PDE is known.
REAL_SMOOTHNESS_WEIGHT = 1e-3

# Optional ablation runs are intentionally shorter than the main run so they can
# produce a dissertation table on a laptop without taking all night.
ABLATION_EPOCHS = 700

# Multi-seed repeats are shorter than the main training run by default. They are
# meant to quantify robustness, not to replace the best full-length model.
MULTI_SEED_VALUES = [7, 42, 123]
MULTI_SEED_EPOCHS = 700

# -----------------------------------------------------------------------------
# Real seismic data settings
# -----------------------------------------------------------------------------
REAL_DATA_DIR = "data/real"
REAL_SECTION_NPY = "data/real/real_section.npy"
REAL_SECTION_CSV = "data/real/real_section.csv"
REAL_SECTION_SGY = "data/real/real_section.sgy"
REAL_SECTION_SEGY = "data/real/real_section.segy"

# Real-data validation diagnostics use metadata when available and otherwise
# clearly report that the real section supports reconstruction claims only.
REAL_VALIDATION_WINDOW = 5

# -----------------------------------------------------------------------------
# Output paths and experiment versioning
# -----------------------------------------------------------------------------
FIGURES_DIR = "results/figures"
DATA_DIR = "results/data"
CONFIG_VERSION = "seismic-pinn-defense-v7-robustness-real-validation"
