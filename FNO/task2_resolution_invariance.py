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

def plot_resolution_invariance(resolutions, test_relative_l2):
    plt.figure(figsize=(8, 5))
    plt.plot(resolutions, test_relative_l2, marker='o')
    plt.xlabel('Resolution (Number of Spatial Points)')
    plt.ylabel('Relative L2 Error (%)') 
    plt.xticks(resolutions)
    plt.title('FNO Performance Across Different Resolutions')
    # add a mark on the training resolution
    plt.axvline(x=128, color='r', linestyle='--')
    plt.show()
    
def perform_experiment(config, model_path="fno_1d_model.pth"):
    model = import_model(config, model_path)
    resolutions, test_relative_l2 = test_resolution_invariance(model, config)
    plot_resolution_invariance(resolutions, test_relative_l2)

if __name__ == "__main__":
    class Config:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MODES = 16
        WIDTH = 64

    config = Config()
    print("Using device:", config.DEVICE)
    perform_experiment(config)