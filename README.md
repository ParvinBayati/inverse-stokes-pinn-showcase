# Physics-Informed Neural Network for Inverse Stokes Flow

## Overview

This repository documents a Physics-Informed Neural Network (PINN) framework developed for solving inverse problems governed by the incompressible Stokes equations.

The framework combines sparse observational data with governing physical laws to reconstruct flow quantities such as slip velocity $u_s$ at the active particle surfaces that is hard to measure experimentally, while enforcing consistency with the underlying fluid dynamics.

The methodology is applicable to a variety of low-Reynolds-number flow problems, including microfluidics, soft matter systems, active matter, and interfacial hydrodynamics.

---

## Governing Equations

The model is based on the incompressible Stokes equations

$$
-\nabla p + \mu \nabla^2 \boldsymbol{u} = \boldsymbol{0}
$$

$$
\nabla \cdot \boldsymbol{u} = 0
$$

where

* $\boldsymbol{u}$ denotes the velocity field,
* $p$ denotes the pressure field,
* $\mu$ is the dynamic viscosity.

### Boundary Conditions

Fluid velocity on the active particle surface

$$\boldsymbol{u}|_{S} = \boldsymbol{U} + \boldsymbol{\Omega} \times (\boldsymbol{r}-\boldsymbol{r}_0) + \boldsymbol{u}_s$$

where $\boldsymbol{U}$ and $\boldsymbol{\Omega}$ are the translational and rotational velocities of the active particle and $\boldsymbol{u}_s$ is the slip velocity at the particle surface.

In case there is a solid boundaries present, the fluid velocity follows no slip conditions on the boundaries:

$$\boldsymbol{u}_{wall} = 0$$

---

## PINN Formulation

A neural network is trained to approximate the unknown solution fields.

The loss function combines:

* Data mismatch loss
* PDE residual loss
* Boundary condition loss

$$\mathcal{L} = \mathcal{L}_{Phys} + \lambda_{data} \mathcal{L}_{data}$$

where

$$\mathcal{L}_{Phys} = \mathcal{L}_{PDE} + \mathcal{L}_{BC} + \mathcal{L}_{wall}$$

Automatic differentiation is used to evaluate the derivatives appearing in the governing equations.

---

## Methodology

The workflow consists of:

1. Data preparation and normalization
2. Physics-informed neural network construction
3. Automatic differentiation of PDE residuals
4. Optimization of the composite loss function
5. Reconstruction of velocity and pressure fields
6. Predicting the velocity at the active particle surface
7. Validation against reference solutions

---

## Representative Results

### Network Architecture

<img src="figures/architecture.jpg" width="500" alt="Architecture">

## Training Workflow

```mermaid
flowchart TD

A[Load training data and collocation points]
B[Initialize 8 neural networks]
C[Define loss function (physics + data)]
D[Choose optimizer (Adam / L-BFGS)]
E[Training loop over epochs]
F[Forward pass: predict velocity and pressure]
G[Compute PDE residuals (Stokes equations)]
H[Compute data loss]
I[Total loss]
J[Backpropagation]
K[Optimizer step]
L[Learning rate update]
M[Save model checkpoints]

A --> B --> C --> D --> E
E --> F --> G --> I
F --> H --> I
I --> J --> K --> L --> M --> E
E --> N[Trained PINN model]
```


### Model Description

The PINN framework consists of eight fully-connected neural networks predicting the exterior and interface velocity and pressure fields:

* Exterior flow:

  * (u_{x,out})
  * (u_{y,out})
  * (u_{z,out})
  * (p_{out})

* Interface flow:

  * (u_{x,s})
  * (u_{y,s})
  * (u_{z,s})
  * (p_s)

Each network contains 9 hidden layers with 128 neurons per layer. During training, the model minimizes a composite loss function

$$\mathcal{L} = \mathcal{L}*{phys} + \lambda*{data}\mathcal{L}_{data}, $$

where the physics loss enforces the governing inverse Stokes equations and the data loss enforces agreement with measured velocity data.


### Loss Function Components

![Loss Function](figures/loss_function.png)

### Predicted Velocity Field

![Velocity Prediction](figures/velocity_prediction.png)

### Predicted Pressure Field

![Pressure Prediction](figures/pressure_prediction.png)

### Convergence History

![Convergence](figures/convergence_history.png)

---

## Publication

The scientific results associated with this work are available in:

https://arxiv.org/abs/2511.22723

---

## Code Availability

The implementation is not publicly distributed. This repository documents the methodology, architecture, and representative results of the PINN framework.

---

## Research Areas

* Physics-Informed Neural Networks (PINNs)
* Scientific Machine Learning
* Inverse Problems
* Stokes Flow
* Computational Fluid Dynamics
* Soft Matter Physics
* Active Matter
* Scientific Computing
