import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

from FNO import FNO1d
torch.manual_seed(0)
np.random.seed(0)

def import_model(config, model_path):
    model = FNO1d(modes=config.MODES, width=config.WIDTH).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path))
    return model

def create_test_dataloaders(config, resolution):
    test = torch.from_numpy(np.load(f"data/data_test_{resolution}.npy")).type(torch.float32)
    u0 = test[:, 0, :]
    u1 = test[:, -1, :]
    u0_grid = torch.cat([u0.unsqueeze(-1), torch.linspace(0, 1, resolution).reshape(1, resolution, 1).repeat(u0.shape[0], 1, 1)], dim=-1)
    u0_grid = u0_grid.to(config.DEVICE)
    u1 = u1.to(config.DEVICE)
    test_set = DataLoader(TensorDataset(u0_grid, u1),  shuffle=False)
    return test_set

def test_resolution_invariance(model, config):
    resolutions = [128, 96, 64, 32]
    test_loaders = [create_test_dataloaders(config, res) for res in resolutions]
    test_relative_l2 = []
    model.eval()
    with torch.no_grad():
        for res, test_loader in zip(resolutions, test_loaders):
            relative_l2 = 0.0
            for input_batch, output_batch in test_loader:
                output_pred_batch = model(input_batch).squeeze(2)
                loss_f = (torch.mean((output_pred_batch - output_batch) ** 2) / torch.mean(output_batch ** 2)) ** 0.5 * 100
                relative_l2 += loss_f.item()
            relative_l2 /= len(test_loader)
            test_relative_l2.append(relative_l2)
            print(f"Resolution {res}: Relative L2 = {relative_l2:.2f}%")
    return resolutions, test_relative_l2

def plot_resolution_invariance_comparison(resolutions, test_relative_l2_12, test_relative_l2_16):
    plt.figure(figsize=(10, 6))
    plt.plot(resolutions, test_relative_l2_12, marker='o', label='12 Modes FNO', linewidth=2)
    plt.plot(resolutions, test_relative_l2_16, marker='s', label='16 Modes FNO', linewidth=2)
    plt.axvline(x=128, color='r', linestyle='--', linewidth=2, label='Training Resolution')
    plt.xlabel('Resolution (Number of Spatial Points)', fontsize=12)
    plt.ylabel('Relative L2 Error (%)', fontsize=12) 
    plt.xticks(resolutions)
    plt.title('FNO Performance Across Different Resolutions: 12 vs 16 Modes', fontweight="bold", fontsize=14)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def perform_experiment(config):
    model_path = f"models/fno_{config.MODES}_model.pth"
    model = import_model(config, model_path)
    resolutions, test_relative_l2 = test_resolution_invariance(model, config)
    return resolutions, test_relative_l2

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODES = 16
    WIDTH = 64

def main():
    config = Config()
    print("Using device:", config.DEVICE)
    
    # Test 12-modes FNO
    print("\n=== Testing 12 Modes FNO ===")
    config.MODES = 12
    resolutions, test_relative_l2_12 = perform_experiment(config)
    
    # Test 16-modes FNO
    print("\n=== Testing 16 Modes FNO ===")
    config.MODES = 16
    resolutions, test_relative_l2_16 = perform_experiment(config)
    
    # Plot comparison
    plot_resolution_invariance_comparison(resolutions, test_relative_l2_12, test_relative_l2_16)

if __name__ == "__main__":
    main()