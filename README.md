

# Physics-Informed Neural Networks for Seismic Wave Propagation in Heterogeneous Media

This project implements a Physics-Informed Neural Network (PINN) framework for modelling two-dimensional seismic wave propagation in a heterogeneous medium. The main experiment compares a PINN approximation against a finite-difference method (FDM) reference solution for the acoustic wave equation.

Project topic:

```text
Физически-информированные нейронные сети для моделирования распространения сейсмических волн в неоднородной среде
```

## 1. Scientific objective

The objective of this project is to investigate whether a neural network constrained by the governing wave equation can approximate seismic wavefields in a heterogeneous medium. The synthetic experiment uses an FDM solver as a numerical reference and trains a PINN using a combined loss function that includes data mismatch, PDE residual, initial condition, and boundary condition terms.

The main equation used in the synthetic experiment is the two-dimensional acoustic wave equation:

```text
u_tt(x,z,t) - c(x,z)^2 [u_xx(x,z,t) + u_zz(x,z,t)] - s(x,z,t) = 0
```

where:

```text
u(x,z,t)   - seismic displacement or pressure field
c(x,z)     - heterogeneous wave velocity model
s(x,z,t)   - seismic source term
x, z       - spatial coordinates
t          - time
```

## 2. Project structure

```text
seismic-pinn-project/
├── data/
│   └── real/
│       └── real_section.sgy
├── results/
│   ├── data/
│   └── figures/
├── scripts/
│   └── prepare_real_section.py
├── src/
│   ├── config.py
│   ├── fdm_solver.py
│   ├── metrics.py
│   ├── pinn_model.py
│   ├── plots.py
│   └── train_pinn.py
├── main.py
├── README.md
└── requirements.txt
```

## 3. Main components

### `src/config.py`

Contains experiment parameters:

```text
spatial and temporal domain
FDM grid size
source parameters
PINN architecture
training settings
loss weights
output paths
real-data paths
```

This file should be adjusted when running stronger or faster experiments.

### `src/fdm_solver.py`

Generates the synthetic reference solution using a second-order finite-difference approximation of the acoustic wave equation. It includes:

```text
heterogeneous velocity model
Ricker wavelet source
CFL stability check
absorbing boundary mask
wavefield simulation
```

The FDM result is treated as the reference solution for quantitative comparison.

### `src/pinn_model.py`

Defines the neural network architecture. The model supports:

```text
Fourier feature encoding
configurable depth and hidden dimension
tanh, sine, or GELU activation
scalar wavefield output u(x,z,t)
```

Fourier features are used to improve the representation of oscillatory wave patterns.

### `src/train_pinn.py`

Trains the PINN using a combined objective:

```text
L = λ_pde L_pde + λ_data L_data + λ_ic L_ic + λ_bc L_bc
```

where:

```text
L_pde   - residual of the acoustic wave equation
L_data  - mismatch between PINN and FDM samples
L_ic    - initial-condition loss
L_bc    - boundary-condition loss
```

The training procedure samples collocation points, FDM data points, initial points, and boundary points.

### `src/metrics.py`

Computes quantitative metrics:

```text
MSE
MAE
Relative L2 Error
Normalized RMSE
PSNR in dB
Correlation coefficient
maximum absolute reference/prediction/error amplitudes
```

These metrics are saved to CSV files and can be used directly in the dissertation result tables.

### `src/plots.py`

Generates publication-style figures:

```text
velocity model
FDM wave snapshots
PINN vs FDM comparison
absolute error maps
training loss curves
```

The FDM and PINN comparison plots use the same symmetric color scale to make visual comparison fair.

## 4. Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch is not installed correctly, install it separately according to the official PyTorch instructions for your system.

## 5. Running the synthetic experiment

The synthetic experiment is the main scientifically controlled experiment for the dissertation.

Run:

```bash
python main.py --mode synthetic
```

This will:

```text
1. create the heterogeneous velocity model
2. solve the acoustic wave equation using FDM
3. train the PINN using FDM samples and PDE residuals
4. generate comparison figures
5. compute quantitative metrics
6. save results to the results/ directory
```

Expected outputs:

```text
results/figures/velocity_model.png
results/figures/fdm_snapshot_*.png
results/figures/comparison_t_*.png
results/figures/loss_curve.png
results/data/metrics.csv
results/data/summary.csv
results/data/pinn_model.pt
```

## 6. Running the real-data reconstruction experiment

The real-data mode uses a real seismic section from:

```text
data/real/real_section.sgy
```

Run:

```bash
python main.py --mode real
```

This mode should be interpreted carefully. Unless a physically consistent velocity model and source are known for the real seismic section, this experiment should be described as:

```text
PINN-based reconstruction of a real seismic section with smoothness or physics-inspired regularization
```

It should not be presented as a full physical simulation of the real subsurface.

Expected outputs:

```text
results/figures/real_section.png
results/figures/real_pinn_reconstruction.png
results/figures/real_loss_curve.png
results/data/summary.csv
```

## 7. Recommended dissertation interpretation

The synthetic experiment should be the main result because the governing equation, velocity model, source term, and reference solution are fully controlled. It allows a direct comparison between the PINN approximation and the FDM reference wavefield.

The real-data experiment should be presented as an additional demonstration. It shows that the same neural-network framework can reconstruct a seismic section, but it does not prove full physical wave propagation unless the corresponding acquisition geometry, source function, boundary conditions, and velocity model are available.

A suitable wording for the dissertation is:

```text
The synthetic experiment demonstrates the ability of a physics-informed neural network to approximate wave propagation in a heterogeneous medium under controlled physical assumptions. The finite-difference solution is used as the reference solution, while the PINN is trained using both supervised wavefield samples and the residual of the acoustic wave equation. The real-data experiment is included as an additional reconstruction example and is interpreted as regularized seismic-section approximation rather than a fully constrained physical simulation.
```

## 8. Important configuration parameters

For faster testing, reduce these values in `src/config.py`:

```python
NX = 60
NZ = 60
NT = 180
N_COLLOCATION = 2000
N_DATA = 8000
EPOCHS = 1000
```

For stronger final dissertation results, increase gradually:

```python
NX = 100
NZ = 100
NT = 300
N_COLLOCATION = 10000
N_DATA = 40000
EPOCHS = 5000
```

Do not increase everything at once on a laptop. First verify that the FDM solver is stable and that the loss curve decreases.

## 9. Result files for dissertation tables

Use:

```text
results/data/metrics.csv
results/data/summary.csv
```

Recommended table columns:

```text
Time
MSE
MAE
Relative L2 Error
NRMSE
PSNR dB
Correlation
```

For the final text, the most understandable metrics are usually:

```text
Relative L2 Error
NRMSE
Correlation
PSNR dB
```

## 10. Common issues

### The FDM solution looks too faint

Because the source term is physically scaled by `dt²`, the amplitude may become small. Increase the source amplitude in `src/config.py`:

```python
SOURCE_AMPLITUDE = 5000.0
```

or use automatic normalization only for visualization.

### Training is too slow

Reduce:

```python
EPOCHS
N_COLLOCATION
N_DATA
HIDDEN_DIM
NETWORK_DEPTH
```

### PDE loss is too large

The explicit source term can make the PDE residual large. Reduce:

```python
LAMBDA_PDE
```

For example:

```python
LAMBDA_PDE = 0.01
```

### PINN prediction is too smooth

Increase Fourier feature strength or model capacity:

```python
FOURIER_MAPPING_SIZE = 128
FOURIER_SCALE = 8.0
HIDDEN_DIM = 192
NETWORK_DEPTH = 8
```

### FDM stability error appears

The CFL condition failed. Increase `NT` or reduce spatial resolution:

```python
NT = 300
```

or:

```python
NX = 60
NZ = 60
```

## 11. Suggested dissertation figures

Recommended figures:

```text
1. Heterogeneous velocity model
2. FDM wavefield snapshots at several times
3. PINN prediction vs FDM reference vs absolute error
4. PINN training loss components
5. Quantitative metrics table
6. Optional real seismic section reconstruction
```

## 12. Methodology summary

The methodology consists of four stages:

```text
1. Define a heterogeneous velocity model c(x,z).
2. Generate a reference wavefield using the finite-difference method.
3. Train a PINN using a physics-informed loss function.
4. Evaluate the PINN against the FDM reference using visual and quantitative metrics.
```

The PINN receives normalized coordinates `(x,z,t)` as input and predicts the scalar wavefield value `u(x,z,t)`. The physical constraint is imposed by differentiating the neural network output with respect to input coordinates using automatic differentiation.