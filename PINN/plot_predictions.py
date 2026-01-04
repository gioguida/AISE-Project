import numpy as np
import matplotlib.pyplot as plt
import torch
from task1_data_generation import Poisson_data_generator
from task2_implementation import PINN, DataDrivenModel

# Configuration
N_HIDDEN_LAYERS = 4
WIDTH = 128
DD_SCALE_FACTOR = 100.0
N_PLOT = 256  # Resolution for plotting
DEVICE = torch.device("cpu")

def load_models(K):
    # PINN
    pinn = PINN(N_HIDDEN_LAYERS, WIDTH, N=64, device=DEVICE, mesh="grid")
    pinn.load_state_dict(torch.load(f'results/models/pinn_K{K}.pt', map_location=DEVICE))
    pinn.eval()
    
    # Data Driven
    dd = DataDrivenModel(N_HIDDEN_LAYERS, WIDTH)
    dd.load_state_dict(torch.load(f'results/models/data_driven_K{K}.pt', map_location=DEVICE))
    dd.eval()
    
    return pinn, dd

def get_predictions(K):
    # Data Generator
    # Note: task2_implementation uses seed 0 for training data generation
    # We use the same seed to get the same exact solution
    generator = Poisson_data_generator(N_PLOT, K, random_seed=0)
    _, sol = generator.generate()
    
    # Grid for predictions
    x = np.linspace(0, 1, N_PLOT)
    y = np.linspace(0, 1, N_PLOT)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    coords = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1),
        dtype=torch.float32,
        device=DEVICE
    )
    
    # Load models
    pinn, dd = load_models(K)
    
    # Predict
    with torch.no_grad():
        pred_pinn = pinn(coords).detach().cpu().numpy().reshape(N_PLOT, N_PLOT)
        pred_dd = dd(coords).detach().cpu().numpy().reshape(N_PLOT, N_PLOT)
    
    # Scale DD prediction
    pred_dd = pred_dd / DD_SCALE_FACTOR
    
    return sol, pred_pinn, pred_dd

def main():
    Ks = [1, 4, 16]
    
    # Create figure
    # 3 rows (Exact, PINN, DD), 3 columns (K=1, 4, 16)
    # This gives a grid where columns are complexities and rows are methods
    # To make it "horizontal shape", we can adjust figsize
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    
    methods = ["Exact Solution", "PINN Prediction", "Data-Driven Prediction"]
    
    for j, K in enumerate(Ks):
        sol, pred_pinn, pred_dd = get_predictions(K)
        
        data_list = [sol, pred_pinn, pred_dd]
        
        for i, data in enumerate(data_list):
            ax = axs[i, j]
            # Use same vmin/vmax for each column (each K) to make comparison valid
            vmin = np.min(sol)
            vmax = np.max(sol)
            
            im = ax.imshow(data.T, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Set titles only for the top row
            if i == 0:
                ax.set_title(f'K = {K}', fontweight='bold', fontsize=18)
            
            # Set row labels
            if j == 0:
                ax.set_ylabel(methods[i], fontweight='bold', fontsize=18)
            
            # Remove ticks to save space
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar for each plot? Or one per column?
            # One per column might be cleaner if they share the scale
            # But let's add a small one to each to be safe
            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('results/plots/comparison_graph.png', dpi=300, bbox_inches='tight')
    print("Plot saved to results/plots/comparison_graph.png")
    plt.show()

if __name__ == "__main__":
    main()
