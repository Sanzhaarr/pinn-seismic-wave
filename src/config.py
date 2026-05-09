import torch

# -----------------------------------------------------------------------------
# Global reproducibility and device settings
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# -----------------------------------------------------------------------------
# Seismic source settings
# -----------------------------------------------------------------------------
SOURCE_X = 0.5
SOURCE_Z = 0.5
SOURCE_FREQUENCY = 8.0
SOURCE_T0 = 0.15
SOURCE_AMPLITUDE = 5000.0

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
N_COLLOCATION = 3000
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
# Diagnostic run: first verify that the neural model can fit the FDM wavefield.
# After the data-only model works, reintroduce physics with a very small value,
# for example LAMBDA_PDE = 1e-6 or 1e-5.
LAMBDA_PDE = 0.0
LAMBDA_DATA = 100.0
LAMBDA_IC = 0.5
LAMBDA_BC = 0.5
LAMBDA_AMPLITUDE = 0.01

# -----------------------------------------------------------------------------
# Real seismic data settings
# -----------------------------------------------------------------------------
REAL_DATA_DIR = "data/real"
REAL_SECTION_NPY = "data/real/real_section.npy"
REAL_SECTION_CSV = "data/real/real_section.csv"
REAL_SECTION_SGY = "data/real/real_section.sgy"
REAL_SECTION_SEGY = "data/real/real_section.segy"

# -----------------------------------------------------------------------------
# Output paths and experiment versioning
# -----------------------------------------------------------------------------
FIGURES_DIR = "results/figures"
DATA_DIR = "results/data"
CONFIG_VERSION = "seismic-pinn-defense-v4-stable-data-fit"