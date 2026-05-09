

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
grid-consistent point-source scaling
CFL stability check
absorbing boundary mask
wavefield simulation
homogeneous, layered, and coarse-grid FDM baselines
sparse-grid interpolation baseline
second faulted heterogeneous validation scenario
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

Run the second heterogeneous scenario:

```bash
python main.py --mode synthetic --scenario faulted
```

Run both synthetic scenarios and save combined scenario tables:

```bash
python main.py --mode synthetic --scenario all
```

For the dissertation ablation table, run:

```bash
python main.py --mode synthetic --ablation
```

For robustness statistics across random initializations, run:

```bash
python main.py --mode synthetic --multi-seed
```

For the strongest but slowest synthetic validation:

```bash
python main.py --mode synthetic --scenario all --ablation --multi-seed
```

This will:

```text
1. create the heterogeneous velocity model
2. solve the acoustic wave equation using FDM
3. run traditional homogeneous, layered, coarse-grid FDM, and sparse interpolation baselines
4. train the weak-PDE PINN using FDM samples and PDE residuals
5. generate comparison figures
6. compute quantitative metrics
7. save results to the results/ directory
```

Expected outputs:

```text
results/figures/velocity_model.png
results/figures/fdm_snapshot_*.png
results/figures/comparison_t_*.png
results/figures/coarse_baseline_comparison_t_*.png
results/figures/sparse_interpolation_comparison_t_*.png
results/figures/loss_curve.png
results/data/metrics.csv
results/data/model_comparison_summary.csv
results/data/coarse_baseline_metrics.csv
results/data/sparse_interpolation_baseline_metrics.csv
results/data/synthetic_scenario_summary.csv        # when --scenario all is used
results/data/model_comparison_scenario_summary.csv # when --scenario all is used
results/data/multi_seed_summary.csv                # when --multi-seed is used
results/data/multi_seed_aggregate.csv              # when --multi-seed is used
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
results/figures/real_section_reference.png
results/figures/real_section_comparison.png
results/figures/real_loss_curve.png
results/figures/real_validation_spectrum.png
results/data/real_metrics.csv
results/data/real_physical_validation.csv
results/data/real_physical_validation_report.md
results/data/real_summary.csv
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
COARSE_NX = 50
COARSE_NZ = 50
COARSE_NT = NT
SPARSE_BASELINE_TIME_STRIDE = 4
SPARSE_BASELINE_SPACE_STRIDE = 4
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
results/data/model_comparison_summary.csv
results/data/homogeneous_baseline_metrics.csv
results/data/layered_baseline_metrics.csv
results/data/coarse_baseline_metrics.csv
results/data/sparse_interpolation_baseline_metrics.csv
results/data/ablation_summary.csv
results/data/synthetic_scenario_summary.csv
results/data/model_comparison_scenario_summary.csv
results/data/multi_seed_summary.csv
results/data/multi_seed_aggregate.csv
results/data/real_physical_validation.csv
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

The point source is scaled by the cell area, so `SOURCE_AMPLITUDE` can stay order-one. If a figure is visually too faint, prefer improving plot limits or increasing the source amplitude gradually:

```python
SOURCE_AMPLITUDE = 1.5
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

The PDE residual can be numerically large because the network uses Fourier features and normalized amplitudes. Keep the weight small:

```python
LAMBDA_PDE
```

For example:

```python
LAMBDA_PDE = 1e-8
```

The training code avoids the immediate point-source neighborhood when sampling PDE collocation points, because the residual is enforced as the homogeneous wave equation away from the source.

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
4. Homogeneous/layered/coarse-grid FDM and sparse interpolation baseline comparisons
5. PINN training loss components
6. Ablation table showing PDE/Fourier/weight sensitivity
7. Faulted-scenario robustness comparison
8. Multi-seed mean/std robustness table
9. Quantitative metrics and model comparison tables
10. Optional real seismic section reconstruction and validation-level report
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
