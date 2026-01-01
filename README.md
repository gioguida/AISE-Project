# AISE Project: Neural Operators and Physics-Informed Neural Networks

This repository contains the code and experiments for the AI for Science and Engineering (AISE) project. The project explores different neural operator architectures and physics-informed learning methods for solving Partial Differential Equations (PDEs).

The repository is organized into three main folders, each focusing on a specific methodology:

## 1. FNO (Fourier Neural Operator)

This folder contains an implementation of the Fourier Neural Operator (FNO) for solving 1D PDEs. It explores various capabilities of FNOs through four distinct tasks.

- **Task 1 (One-to-One)**: Training an FNO to map initial conditions to solutions at a fixed time.
- **Task 2 (Resolution Invariance)**: Verifying the resolution invariance property of the trained FNO model.
- **Task 3 (All-to-All)**: Autoregressive modeling to map solutions from time $t$ to $t+1$.
- **Task 4 (Fine-tuning)**: Evaluating performance on out-of-distribution data and comparing zero-shot, fine-tuning, and training from scratch.

**Key Files:**
- `FNO.py`, `FNO_bn.py`: Model definitions.
- `run_all_tasks.py`: Main script to run experiments.

*For detailed instructions, see [FNO/README.md](FNO/README.md).*

## 2. GAOT (Geometry-Aware Operator Transformer)

This folder implements the Geometry-Aware Operator Transformer (GAOT/GOAT) and applies it to elasticity problems.

- **Task 1 & Task 2**: Experiments using the GOAT model on elasticity datasets.
- **Config**: Configuration files (JSON/TOML) for different experiment scales (`elasticity_big`, `elasticity_fast`).
- **Src**: Source code for the model, trainer, and dataset handling.

## 3. PINN (Physics-Informed Neural Networks)

This folder investigates the performance and failure modes of Physics-Informed Neural Networks (PINNs) compared to purely Data-Driven models, with a focus on loss landscape analysis.

- **Task 1**: Data generation for the Poisson equation.
- **Task 2**: Implementation and training of PINN and Data-Driven models.
- **Task 3**: Visualization of 3D loss landscapes to analyze convergence and optimization difficulties.

**Key Files:**
- `experiment_runner.py`: Main script for training and evaluation.
- `task3_loss_landscape.py`: Script for generating loss landscape plots.

*For detailed instructions, see [PINN/README.md](PINN/README.md).*

## Requirements

Please refer to the `requirements.txt` file in the root directory (if available) or the individual READMEs in each subfolder for specific dependencies. Generally, the project requires:
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- SciPy
