import torch

# -----------------------------------------------------------------------------
# Global reproducibility and device settings
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Computational domain
# -----------------------------------------------------------------------------
# The synthetic experiment uses normalized coordinates. This makes the PINN
# training numerically more stable and keeps derivatives on a comparable scale.
X_MIN, X_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 0.6

# -----------------------------------------------------------------------------
# Finite-difference reference solver
# -----------------------------------------------------------------------------
# These values are intentionally moderate so the experiment can run on a laptop.
# For final dissertation figures, increase NX/NZ/NT gradually and compare runtime.
NX = 80
NZ = 80
NT = 240

# -----------------------------------------------------------------------------
# Seismic source settings
# -----------------------------------------------------------------------------
SOURCE_X = 0.5
SOURCE_Z = 0.5
SOURCE_FREQUENCY = 8.0
SOURCE_T0 = 0.15
SOURCE_AMPLITUDE = 80.0

# -----------------------------------------------------------------------------
# PINN architecture defaults
# -----------------------------------------------------------------------------
USE_FOURIER_FEATURES = True
FOURIER_MAPPING_SIZE = 128
FOURIER_SCALE = 8.0
HIDDEN_DIM = 192
NETWORK_DEPTH = 8
ACTIVATION = "tanh"  # options supported by SeismicPINN: "tanh", "sine", "gelu"

# -----------------------------------------------------------------------------
# PINN training settings
# -----------------------------------------------------------------------------
# Collocation points enforce the wave equation.
# Data points anchor the PINN to the finite-difference reference solution.
# Initial and boundary points stabilize the solution at t=0 and domain edges.
N_COLLOCATION = 6000
N_DATA = 24000
N_INITIAL = 2000
N_BOUNDARY = 2000

EPOCHS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0

# Optional second-stage optimizer. LBFGS often improves PINN convergence after
# Adam, but it is slower. Keep it disabled for quick experiments.
USE_LBFGS_REFINEMENT = False
LBFGS_MAX_ITER = 300

# -----------------------------------------------------------------------------
# Loss weights
# -----------------------------------------------------------------------------
# Dissertation-oriented setting: keep the data term strong enough for agreement
# with FDM, but make the PDE residual meaningful enough to justify the PINN setup.
LAMBDA_PDE = 0.2
LAMBDA_DATA = 80.0
LAMBDA_IC = 5.0
LAMBDA_BC = 2.0

# -----------------------------------------------------------------------------
# Real seismic data settings
# -----------------------------------------------------------------------------
# Real data mode should be described as reconstruction with smoothness/physics
# regularization unless a physically consistent source and velocity model are used.
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
CONFIG_VERSION = "seismic-pinn-dissertation-v1"