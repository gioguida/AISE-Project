
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from task2_implementation import (
    Config, 
    Poisson_data_generator,
    evaluate_model,
    PINN,
    DataDrivenModel
)

class RecreateConfig:
    K_LEVELS = [1, 4, 16]
    K_LABELS = {1: "Low", 4: "Medium", 16: "High"}
    N = 64
    OUTPUT_DIR = "results"
    PLOTS_DIR = "results/plots"
    MODELS_DIR = "results/models"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def recreate_comparison_plots():
    print("Recreating comparison plots using saved models...")
    
    os.makedirs(RecreateConfig.PLOTS_DIR, exist_ok=True)
    
    config = Config()
    config.N = RecreateConfig.N
    config.DEVICE = RecreateConfig.DEVICE
    
    for K in RecreateConfig.K_LEVELS:
        complexity_label = RecreateConfig.K_LABELS[K]
        print(f"\nProcessing K={K} ({complexity_label})...")
        
        config.K = K
        
        # Generate data
        data_generator = Poisson_data_generator(config.N, config.K)
        force, solution = data_generator.generate()
        
        # Load PINN
        pinn_path = os.path.join(RecreateConfig.MODELS_DIR, f'pinn_K{K}.pt')
        if os.path.exists(pinn_path):
            print(f"  Loading PINN from {pinn_path}")
            pinn_model = PINN(
                config.N_HIDDEN_LAYERS, 
                config.WIDTH, 
                config.N, 
                config.DEVICE,
                mesh=config.MESH_TYPE,
                lambda_u=config.PINN_LAMBDA_U
            )
            pinn_model.load_state_dict(torch.load(pinn_path, map_location=config.DEVICE))
            
            U_pred_pinn, U_exact, error_pinn = evaluate_model(
                pinn_model, data_generator, config, is_data_driven=False
            )
            print(f"  PINN Error: {error_pinn:.2f}%")
        else:
            print(f"  Warning: PINN model not found at {pinn_path}")
            continue

        # Load Data-Driven
        dd_path = os.path.join(RecreateConfig.MODELS_DIR, f'data_driven_K{K}.pt')
        if os.path.exists(dd_path):
            print(f"  Loading Data-Driven from {dd_path}")
            dd_model = DataDrivenModel(config.N_HIDDEN_LAYERS, config.WIDTH).to(config.DEVICE)
            dd_model.load_state_dict(torch.load(dd_path, map_location=config.DEVICE))
            
            U_pred_dd, _, error_dd = evaluate_model(
                dd_model, data_generator, config, is_data_driven=True
            )
            print(f"  Data-Driven Error: {error_dd:.2f}%")
        else:
            print(f"  Warning: Data-Driven model not found at {dd_path}")
            continue
            
        # Generate Plot
        print("  Generating comparison plot...")
        x = np.linspace(0, 1, config.N)
        y = np.linspace(0, 1, config.N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Row 1: PINN results
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.pcolormesh(X, Y, U_pred_pinn, shading='auto', cmap='viridis')
        ax1.set_title('PINN Prediction', fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.pcolormesh(X, Y, U_exact, shading='auto', cmap='viridis')
        ax2.set_title('Exact Solution', fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        error_pinn_abs = np.abs(U_pred_pinn - U_exact)
        im3 = ax3.pcolormesh(X, Y, error_pinn_abs, shading='auto', cmap='Reds')
        ax3.set_title(f'PINN Error\nL2 Rel: {error_pinn:.2f}%', fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(im3, ax=ax3)
        
        # Row 2: Data-Driven results
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.pcolormesh(X, Y, U_pred_dd, shading='auto', cmap='viridis')
        ax4.set_title('Data-Driven Prediction', fontweight='bold')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        plt.colorbar(im4, ax=ax4)
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.pcolormesh(X, Y, U_exact, shading='auto', cmap='viridis')
        ax5.set_title('Exact Solution', fontweight='bold')
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
        plt.colorbar(im5, ax=ax5)
        
        ax6 = fig.add_subplot(gs[1, 2])
        error_dd_abs = np.abs(U_pred_dd - U_exact)
        im6 = ax6.pcolormesh(X, Y, error_dd_abs, shading='auto', cmap='Reds')
        ax6.set_title(f'Data-Driven Error\nL2 Rel: {error_dd:.2f}%', fontweight='bold')
        ax6.set_xlabel('x')
        ax6.set_ylabel('y')
        plt.colorbar(im6, ax=ax6)
        
        fig.suptitle(f'Model Comparison - Complexity {complexity_label} (K={K})', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        comp_path = os.path.join(RecreateConfig.PLOTS_DIR, f'comparison_K{K}.png')
        plt.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {comp_path}")

if __name__ == "__main__":
    recreate_comparison_plots()
