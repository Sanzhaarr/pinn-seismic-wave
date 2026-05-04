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
FOURIER_SCALE = 8.0
HIDDEN_DIM = 160
NETWORK_DEPTH = 7
ACTIVATION = "tanh"

# -----------------------------------------------------------------------------
# PINN training settings
# -----------------------------------------------------------------------------
N_COLLOCATION = 3000
N_DATA = 16000
N_INITIAL = 1200
N_BOUNDARY = 1200

EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0

USE_LBFGS_REFINEMENT = False
LBFGS_MAX_ITER = 300

# -----------------------------------------------------------------------------
# Loss weights
# -----------------------------------------------------------------------------
# The FDM wavefield is normalized during training. The PDE term is intentionally
# weak for the defense run so the PINN reconstructs the visible wavefield well
# while still preserving physics-informed regularization.
LAMBDA_PDE = 0.001
LAMBDA_DATA = 100.0
LAMBDA_IC = 2.0
LAMBDA_BC = 1.0

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
CONFIG_VERSION = "seismic-pinn-defense-v2-normalized"