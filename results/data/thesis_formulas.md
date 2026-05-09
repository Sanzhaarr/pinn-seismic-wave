# Thesis Formulas

Use these equations in the methodology and evaluation chapters.

## Acoustic Wave Equation

The controlled synthetic experiments solve the 2D acoustic wave equation in a heterogeneous medium:

```latex
\frac{\partial^2 u(x,z,t)}{\partial t^2}
= c(x,z)^2
\left(
\frac{\partial^2 u(x,z,t)}{\partial x^2}
+ \frac{\partial^2 u(x,z,t)}{\partial z^2}
\right)
+ s(x,z,t)
```

where \(u(x,z,t)\) is the seismic wavefield, \(c(x,z)\) is the spatially varying velocity model, and \(s(x,z,t)\) is the source term.

## PINN Approximation

The neural network approximates the wavefield as:

```latex
\hat{u}_\theta(x,z,t) \approx u(x,z,t)
```

where \(\theta\) denotes the trainable neural-network parameters.

## PDE Residual

Away from the point-source neighborhood, the physics-informed residual is:

```latex
r_\theta(x,z,t)
=
\frac{\partial^2 \hat{u}_\theta}{\partial t^2}
- c(x,z)^2
\left(
\frac{\partial^2 \hat{u}_\theta}{\partial x^2}
+ \frac{\partial^2 \hat{u}_\theta}{\partial z^2}
\right)
```

The residual is evaluated by automatic differentiation.

## Training Objective

The synthetic PINN is trained with a weighted objective:

```latex
\mathcal{L}
=
\lambda_{\mathrm{data}}\mathcal{L}_{\mathrm{data}}
+ \lambda_{\mathrm{pde}}\mathcal{L}_{\mathrm{pde}}
+ \lambda_{\mathrm{ic}}\mathcal{L}_{\mathrm{ic}}
+ \lambda_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}}
+ \lambda_{\mathrm{amp}}\mathcal{L}_{\mathrm{amp}}
```

with:

```latex
\mathcal{L}_{\mathrm{data}}
=
\frac{1}{N_d}
\sum_{i=1}^{N_d}
\left(
\hat{u}_\theta(x_i,z_i,t_i)-u_{\mathrm{FDM}}(x_i,z_i,t_i)
\right)^2
```

```latex
\mathcal{L}_{\mathrm{pde}}
=
\frac{1}{N_c}
\sum_{j=1}^{N_c}
r_\theta(x_j,z_j,t_j)^2
```

The project uses a small nonzero PDE weight. This makes the model a weak-PDE or PDE-regularized PINN: the data term fits the finite-difference wavefield, while the PDE term regularizes the solution toward acoustic-wave physics.

## Ricker Source Wavelet

The synthetic source is a Ricker wavelet:

```latex
s(t)
=
A
\left(
1 - 2\pi^2 f_0^2 (t-t_0)^2
\right)
\exp\left(
-\pi^2 f_0^2 (t-t_0)^2
\right)
```

For the final multi-source experiment, the total source is a sum of individual Ricker sources:

```latex
s(x,z,t)
=
\sum_{k=1}^{N_s}
A_k
\left(
1 - 2\pi^2 f_k^2 (t-t_{0,k})^2
\right)
\exp\left(
-\pi^2 f_k^2 (t-t_{0,k})^2
\right)
\delta(x-x_k)\delta(z-z_k)
```

## Finite-Difference Update

The reference wavefield is generated with a second-order finite-difference update:

```latex
u_{i,j}^{n+1}
=
2u_{i,j}^{n}
- u_{i,j}^{n-1}
+ \Delta t^2 c_{i,j}^2
\left[
\frac{u_{i+1,j}^{n}-2u_{i,j}^{n}+u_{i-1,j}^{n}}{\Delta x^2}
+
\frac{u_{i,j+1}^{n}-2u_{i,j}^{n}+u_{i,j-1}^{n}}{\Delta z^2}
\right]
+ \Delta t^2 s_{i,j}^{n}
```

## Stability Condition

The CFL stability condition used for the 2D FDM scheme is:

```latex
\mathrm{CFL}
=
c_{\max}\Delta t
\sqrt{
\frac{1}{\Delta x^2}
+
\frac{1}{\Delta z^2}
}
< 1
```

## Evaluation Metrics

Mean squared error:

```latex
\mathrm{MSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{u}_i-u_i)^2
```

Mean absolute error:

```latex
\mathrm{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
|\hat{u}_i-u_i|
```

Relative L2 error:

```latex
\mathrm{Relative\ L2}
=
\frac{\|\hat{u}-u\|_2}{\|u\|_2}
```

Normalized RMSE:

```latex
\mathrm{NRMSE}
=
\frac{
\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{u}_i-u_i)^2}
}{
\max_i |u_i|
}
```

Correlation coefficient:

```latex
\rho
=
\frac{
\sum_i(\hat{u}_i-\bar{\hat{u}})(u_i-\bar{u})
}{
\sqrt{\sum_i(\hat{u}_i-\bar{\hat{u}})^2}
\sqrt{\sum_i(u_i-\bar{u})^2}
}
```

Energy ratio:

```latex
E_{\mathrm{ratio}}
=
\frac{\sum_i \hat{u}_i^2}{\sum_i u_i^2}
```

## Recommended Scope Statement

```text
This dissertation investigates physics-informed neural networks for 2D acoustic seismic wave propagation in heterogeneous media. The main validation is performed on controlled synthetic experiments where the velocity model, source, boundary treatment, and finite-difference reference solution are known. Real seismic data are used as an additional reconstruction demonstration rather than as full proof of physical field-scale wave propagation.
```
