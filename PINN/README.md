# PINN vs Data-Driven: Loss Landscape Analysis

This project investigates the failure modes of Physics-Informed Neural Networks (PINNs) compared to Data-Driven models by analyzing their loss landscapes across different complexity levels (frequency modes K).

## Project Structure

- `experiment_runner.py`: Main script to run experiments, train models, and generate performance comparison plots.
- `task3_loss_landscape.py`: Script to compute and visualize the 3D loss landscapes of the trained models.
- `task1_data_generation.py`: Module for generating the Poisson equation data.
- `task2_implementation.py`: Core implementation of PINN and Data-Driven models and training loops.
- `results/`: Directory where all models, plots, and data are saved.

## 1. Running Experiments (`experiment_runner.py`)

This script handles the end-to-end pipeline: generating data, training models (PINN and Data-Driven), and evaluating their performance.

### Usage
```bash
python experiment_runner.py
```

### Configuration
You can modify the `ExperimentConfig` class inside `experiment_runner.py` to control the execution:

```python
class ExperimentConfig:
    # ...
    
    # Complexity levels to test
    K_LEVELS = [1, 4, 16]
    
    # Execution Flags
    TRAIN_PINN = True          # Set to False to skip PINN training/evaluation
    TRAIN_DD = True            # Set to False to skip Data-Driven training/evaluation
    
    RETRAIN = False            # If True, forces retraining even if models exist
    
    RECREATE_PLOTS_ONLY = False # If True, skips training and only generates plots using saved models
```

**Common Workflows:**
- **Run everything:** Keep defaults (`TRAIN_PINN=True`, `TRAIN_DD=True`, `RETRAIN=False`).
- **Re-generate plots only:** Set `RECREATE_PLOTS_ONLY = True`.
- **Retrain only Data-Driven models:** Set `TRAIN_PINN = False`, `TRAIN_DD = True`, `RETRAIN = True`.

## 2. Visualizing Loss Landscapes (`task3_loss_landscape.py`)

This script generates 3D surface plots of the loss landscape around the converged solution. It uses the filter-normalization method to visualize the geometry of the loss function.

### Usage
```bash
python task3_loss_landscape.py
```

### Configuration
Modify the `LandscapeConfig` class inside `task3_loss_landscape.py`:

```python
class LandscapeConfig:
    # ...
    
    # Plotting Limits (Z-axis)
    VMAX_LEVELS = [10, 100]    # Generates plots with these max loss heights
    
    # Models to process
    K_LEVELS = [1, 4, 16]
    MODELS = ['pinn', 'dd']
    
    # Computation
    FORCE_RECOMPUTE = False    # Set to True to re-calculate the loss grid
```

### Important Note on Performance
- The first run for a specific model and K-level is **computationally expensive** as it calculates the loss on a 100x100 grid.
- The computed surface data is saved to `.h5` files in `results/loss_landscapes_adapted/`.
- Subsequent runs with the same grid settings will **load the saved data** and only re-generate the plots. This allows you to quickly adjust plot settings (like `VMAX_LEVELS` or font sizes) without re-computing the losses.

## Requirements
- PyTorch
- NumPy
- Matplotlib
- h5py
- (Optional) mpi4pytorch
