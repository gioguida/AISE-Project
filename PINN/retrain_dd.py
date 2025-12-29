
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import json
from datetime import datetime
from task2_implementation import (
    Config, 
    Poisson_data_generator,
    train_data_driven,
    evaluate_model,
    PINN
)

class ExperimentConfig:
    K_LEVELS = [1, 4, 16]
    K_LABELS = {1: "Low", 4: "Medium", 16: "High"}
    N = 64
    OUTPUT_DIR = "results"
    PLOTS_DIR = "results/plots"
    MODELS_DIR = "results/models"
    RESULTS_FILE = "results/retrain_results.txt"
    RESULTS_JSON = "results/experiment_results.json"
    SEED = 42

class ResultLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Retraining Data-Driven Models - {datetime.now()}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

def retrain_dd_only():
    torch.manual_seed(ExperimentConfig.SEED)
    np.random.seed(ExperimentConfig.SEED)
    logger = ResultLogger(ExperimentConfig.RESULTS_FILE)
    
    # Load existing results if available
    existing_results = {}
    if os.path.exists(ExperimentConfig.RESULTS_JSON):
        try:
            with open(ExperimentConfig.RESULTS_JSON, 'r') as f:
                data = json.load(f)
                if 'results' in data:
                    # Convert string keys to int
                    for k, v in data['results'].items():
                        existing_results[int(k)] = v
            logger.log("Loaded existing results from JSON.")
        except Exception as e:
            logger.log(f"Could not load existing JSON results: {e}")

    config = Config()
    config.N = ExperimentConfig.N
    
    for K in ExperimentConfig.K_LEVELS:
        logger.log(f"\n{'='*60}")
        logger.log(f"Processing K={K} ({ExperimentConfig.K_LABELS[K]})")
        logger.log(f"{'='*60}")
        
        config.K = K
        
        # Generate data
        data_generator = Poisson_data_generator(config.N, config.K)
        force, solution = data_generator.generate()
        
        # Try to load PINN for comparison
        pinn_model = None
        U_pred_pinn = None
        error_pinn = 0.0
        pinn_path = os.path.join(ExperimentConfig.MODELS_DIR, f'pinn_K{K}.pt')
        
        if os.path.exists(pinn_path):
            logger.log("Found existing PINN model, loading for comparison...")
            try:
                pinn_model = PINN(config.N_HIDDEN_LAYERS, config.WIDTH, config.N, config.DEVICE, mesh=config.MESH_TYPE, lambda_u=config.PINN_LAMBDA_U)
                pinn_model.load_state_dict(torch.load(pinn_path, map_location=config.DEVICE))
                U_pred_pinn, _, error_pinn = evaluate_model(pinn_model, data_generator, config, is_data_driven=False)
                logger.log(f"PINN Error: {error_pinn:.2f}%")
            except Exception as e:
                logger.log(f"Failed to load PINN model: {e}")
                pinn_model = None
        else:
            logger.log("No existing PINN model found. Skipping PINN comparison.")

        # Train Data-Driven
        logger.log("\nTraining Data-Driven model...")
        dd_model, dd_history = train_data_driven(config, data_generator, verbose=True)
        U_pred_dd, U_exact, error_dd = evaluate_model(dd_model, data_generator, config, is_data_driven=True)
        logger.log(f"Data-Driven Error: {error_dd:.2f}%")
        
        # Save DD Model
        os.makedirs(ExperimentConfig.MODELS_DIR, exist_ok=True)
        torch.save(dd_model.state_dict(), os.path.join(ExperimentConfig.MODELS_DIR, f'data_driven_K{K}.pt'))
        
        # Save DD Loss Plot
        os.makedirs(ExperimentConfig.PLOTS_DIR, exist_ok=True)
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(dd_history, linewidth=1.5, color='#A23B72')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Data-Driven Training History (K={K})')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(ExperimentConfig.PLOTS_DIR, f'data_driven_loss_K{K}.png'), bbox_inches='tight')
        plt.close()

        # Comparison Plot
        if U_pred_pinn is not None:
            logger.log("Generating comparison plots...")
            x = np.linspace(0, 1, config.N)
            y = np.linspace(0, 1, config.N)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Row 1: PINN results
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.pcolormesh(X, Y, U_pred_pinn, shading='auto', cmap='viridis')
            ax1.set_title('PINN Prediction', fontweight='bold')
            plt.colorbar(im1, ax=ax1)
            
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.pcolormesh(X, Y, U_exact, shading='auto', cmap='viridis')
            ax2.set_title('Exact Solution', fontweight='bold')
            plt.colorbar(im2, ax=ax2)
            
            ax3 = fig.add_subplot(gs[0, 2])
            error_pinn_abs = np.abs(U_pred_pinn - U_exact)
            im3 = ax3.pcolormesh(X, Y, error_pinn_abs, shading='auto', cmap='Reds')
            ax3.set_title(f'PINN Error\nL2 Rel: {error_pinn:.2f}%', fontweight='bold')
            plt.colorbar(im3, ax=ax3)
            
            # Row 2: Data-Driven results
            ax4 = fig.add_subplot(gs[1, 0])
            im4 = ax4.pcolormesh(X, Y, U_pred_dd, shading='auto', cmap='viridis')
            ax4.set_title('Data-Driven Prediction', fontweight='bold')
            plt.colorbar(im4, ax=ax4)
            
            ax5 = fig.add_subplot(gs[1, 1])
            im5 = ax5.pcolormesh(X, Y, U_exact, shading='auto', cmap='viridis')
            ax5.set_title('Exact Solution', fontweight='bold')
            plt.colorbar(im5, ax=ax5)
            
            ax6 = fig.add_subplot(gs[1, 2])
            error_dd_abs = np.abs(U_pred_dd - U_exact)
            im6 = ax6.pcolormesh(X, Y, error_dd_abs, shading='auto', cmap='Reds')
            ax6.set_title(f'Data-Driven Error\nL2 Rel: {error_dd:.2f}%', fontweight='bold')
            plt.colorbar(im6, ax=ax6)
            
            fig.suptitle(f'Model Comparison - Complexity {ExperimentConfig.K_LABELS[K]} (K={K})', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            comp_path = os.path.join(ExperimentConfig.PLOTS_DIR, f'comparison_K{K}.png')
            plt.savefig(comp_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Update JSON structure
        if K not in existing_results:
            existing_results[K] = {}
            existing_results[K]['complexity_label'] = ExperimentConfig.K_LABELS[K]
            
        existing_results[K]['data_driven'] = {
            'error': float(error_dd),
            'final_loss': float(dd_history[-1]),
            'iterations': len(dd_history)
        }
        
        # If PINN was loaded and not in results, add it
        if 'pinn' not in existing_results[K] and U_pred_pinn is not None:
             existing_results[K]['pinn'] = {
                'error': float(error_pinn),
                'final_loss': 0.0, # Unknown
                'iterations': 0 # Unknown
            }

    # Save updated JSON
    final_json = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {'N': config.N, 'K_levels': ExperimentConfig.K_LEVELS, 'seed': ExperimentConfig.SEED},
        'results': existing_results
    }
    with open(ExperimentConfig.RESULTS_JSON, 'w') as f:
        json.dump(final_json, f, indent=2)
        
    logger.log(f"\nUpdated results saved to {ExperimentConfig.RESULTS_JSON}")
    logger.log("Retraining completed.")

if __name__ == "__main__":
    retrain_dd_only()
