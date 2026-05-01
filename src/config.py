import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Spatial domain
X_MIN, X_MAX = 0.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 0.6

# Finite difference grid
NX = 80
NZ = 80
NT = 250

# Source position
SOURCE_X = 0.5
SOURCE_Z = 0.5

# PINN training
N_COLLOCATION = 12000
N_DATA = 4000
N_INITIAL = 2000
N_BOUNDARY = 2000

EPOCHS = 3000
LEARNING_RATE = 1e-3

# Loss weights
LAMBDA_PDE = 1.0
LAMBDA_DATA = 10.0
LAMBDA_IC = 5.0
LAMBDA_BC = 2.0