# Fourier Neural Operator (FNO) Project

This repository contains an implementation of the Fourier Neural Operator (FNO) for solving 1D Partial Differential Equations (PDEs). The project is structured into four main tasks, exploring different aspects of FNOs, including one-to-one mapping, resolution invariance, all-to-all mapping, and fine-tuning on new data.

## Project Structure

### Core Files
- **`FNO.py`**: Defines the standard `FNO1d` model and the `SpectralConv1d` layer.
- **`FNO_bn.py`**: Defines `FNO1d_bn`, a variation of the FNO model that includes Batch Normalization and FiLM layers.
- **`run_all_tasks.py`**: The main orchestration script to configure and run all tasks sequentially.

### Task Scripts
- **`task1_one2one.py`**: 
  - **Goal**: Train an FNO to map the initial condition $u(x, 0)$ to the solution at a fixed later time $u(x, T)$.
  - **Model**: Uses `FNO1d`.
- **`task2_resolution_invariance.py`**: 
  - **Goal**: Verify the resolution invariance property of the FNO by testing a trained model on data with different spatial resolutions (128, 96, 64, 32 grid points).
  - **Model**: Uses the model trained in Task 1.
- **`task3_all2all.py`**: 
  - **Goal**: Train an FNO to map the solution at time $t$ to the solution at time $t+1$ (autoregressive modeling).
  - **Model**: Uses `FNO1d_bn`.
- **`task4_finetuning.py`**: 
  - **Goal**: Evaluate the model's performance on a dataset with "unknown" parameters (distribution shift). Compares three approaches:
    1. **Zero-shot**: Using the pre-trained model from Task 3 directly.
    2. **Fine-tuning**: Fine-tuning the pre-trained model on a small amount of new data.
    3. **Scratch**: Training a model from scratch on the new data.

### Directories
- **`data/`**: Contains the dataset files (`.npy`) for training, validation, and testing at various resolutions.
- **`models/`**: Stores the saved PyTorch model weights (`.pth`).
- **`notebooks/`**: Jupyter notebooks corresponding to each task for interactive experimentation and visualization.
- **`results/`**: Contains text files with output logs and results.

## Dependencies

Ensure you have the following Python packages installed:
- `torch`
- `numpy`
- `matplotlib`

## How to Run Experiments

### Option 1: Run All Tasks (Recommended)
You can run all tasks or a selection of them using the `run_all_tasks.py` script. This script uses a `GlobalConfig` class to control the execution flow.

1. Open `run_all_tasks.py`.
2. Modify the `GlobalConfig` class to select which tasks to run and whether to retrain models:
   ```python
   class GlobalConfig:
       # ...
       # Task 1: One-to-One
       TASK1_RETRAIN = False  # Set to True to retrain
       TASK1_SAVE = False     # Set to True to save the new model

       # Task 3: All-to-All
       TASK3_RETRAIN = False
       # ...
   ```
3. Run the script:
   ```bash
   python run_all_tasks.py
   ```

### Option 2: Run Individual Tasks
Each task script can be executed independently. They contain their own `Config` classes for local settings.

- **Task 1**: `python task1_one2one.py`
- **Task 2**: `python task2_resolution_invariance.py`
- **Task 3**: `python task3_all2all.py`
- **Task 4**: `python task4_finetuning.py`

### Option 3: Jupyter Notebooks
Navigate to the `notebooks/` directory to run the interactive notebooks:
- `task1_one_to_one.ipynb`
- `task2_resolution_invariance.ipynb`
- `task3_all2all.ipynb`
- `task4_finetuning.ipynb`

## Model Configuration
The default model configuration (number of modes, width) is defined in the `Config` classes within each script or centrally in `run_all_tasks.py`.
- `MODES`: Number of Fourier modes to keep.
- `WIDTH`: Hidden channel dimension.
