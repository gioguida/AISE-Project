"""
AISE 2026 Final Project - Problem 1: Experiment Runner
Runs all required experiments for PINN vs Data-Driven comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from datetime import datetime
from pathlib import Path
import json

# Import from the refactored code
from task1_data_generation import Poisson_data_generator
from task2_implementation import (
    Config, 
    train_pinn,
    train_data_driven,
    evaluate_model,
    plot_training_history, 
    plot_solution_comparison,
    PINN,
    DataDrivenModel
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

class ExperimentConfig:
    """Configuration for the complete experiment suite"""
    
    # Complexity levels to test (as per project requirements)
    K_LEVELS = [1, 4, 16]  # Low, Medium, High
    K_LABELS = {1: "Low", 4: "Medium", 16: "High"}
    
    # Additional K for visualization (Task 1)
    K_VISUALIZATION = [1, 4, 16]
    
    # Grid resolution
    N = 64
    
    # Output directories
    OUTPUT_DIR = "results"
    PLOTS_DIR = "results/plots"
    MODELS_DIR = "results/models"
    DATA_DIR = "results/data"
    
    # Results file
    RESULTS_FILE = "results/experiment_results.txt"
    RESULTS_JSON = "results/experiment_results.json"
    
    # Random seed for reproducibility
    SEED = 42

    # ========================================================================
    # EXECUTION FLAGS
    # ========================================================================
    
    # Which models to process
    TRAIN_PINN = True
    TRAIN_DD = True
    
    # Execution mode
    # If True, will ignore existing models and retrain from scratch
    RETRAIN = False 
    
    # If True, will skip all training/loading logic that isn't needed for plotting
    # and just try to load models to generate plots.
    # If models are missing, it will skip them.
    RECREATE_PLOTS_ONLY = True


# ============================================================================
# OUTPUT FORMATTING UTILITIES
# ============================================================================

class ResultLogger:
    """Handles formatted output to both terminal and file"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.terminal_output = []
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Initialize file with header
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AISE 2026 - Final Project - Problem 1: PINNs vs Data-Driven\n")
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message, to_file=True, to_terminal=True):
        """Log message to file and/or terminal"""
        if to_terminal:
            print(message)
            self.terminal_output.append(message)
        
        if to_file:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(message + "\n")
    
    def section(self, title, char="="):
        """Create a section header"""
        self.log(f"\n{title}")
    
    def subsection(self, title):
        """Create a subsection header"""
        self.log(f"\n{title}")
    
    def table_header(self, headers, widths):
        """Create formatted table header"""
        header_line = " | ".join([h.center(w) for h, w in zip(headers, widths)])
        separator = "-+-".join(["-"*w for w in widths])
        self.log(header_line)
        self.log(separator)
    
    def table_row(self, values, widths):
        """Create formatted table row"""
        row = " | ".join([str(v).center(w) for v, w in zip(values, widths)])
        self.log(row)


# ============================================================================
# DATA GENERATION (TASK 1)
# ============================================================================

def task1_data_generation(logger, exp_config):
    """
    Task 1: Generate and visualize datasets for different K values
    """
    logger.section("Task 1: Data Generation", "=")
    logger.log(f"K values: {exp_config.K_VISUALIZATION}, N={exp_config.N}")
    
    # Create data directory
    os.makedirs(exp_config.DATA_DIR, exist_ok=True)
    
    # Create figure for visualization
    n_samples = len(exp_config.K_VISUALIZATION)
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    x = np.linspace(0, 1, exp_config.N)
    y = np.linspace(0, 1, exp_config.N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    data_samples = {}
    
    for idx, K in enumerate(exp_config.K_VISUALIZATION):
        logger.subsection(f"Generating sample with K = {K}")
        
        # Generate data
        data_gen = Poisson_data_generator(exp_config.N, K)
        force, solution = data_gen.generate()
        
        # Store sample
        data_samples[K] = {
            'force': force,
            'solution': solution,
            'generator': data_gen
        }
        
        # Plot forcing term
        im0 = axes[0, idx].pcolormesh(X, Y, force, shading='auto', cmap='RdBu_r')
        axes[0, idx].set_title(f'Forcing Term f\n(K={K})', fontweight='bold')
        axes[0, idx].set_xlabel('x')
        axes[0, idx].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, idx])
        
        # Plot solution
        im1 = axes[1, idx].pcolormesh(X, Y, solution, shading='auto', cmap='viridis')
        axes[1, idx].set_title(f'Exact Solution u\n(K={K})', fontweight='bold')
        axes[1, idx].set_xlabel('x')
        axes[1, idx].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1, idx])
        
        # Log statistics
        logger.log(f"  Forcing term - Min: {force.min():.4f}, Max: {force.max():.4f}, Mean: {force.mean():.4f}")
        logger.log(f"  Solution     - Min: {solution.min():.4f}, Max: {solution.max():.4f}, Mean: {solution.mean():.4f}")
    
    plt.tight_layout()
    fig_path = os.path.join(exp_config.PLOTS_DIR, 'task1_data_samples.png')
    os.makedirs(exp_config.PLOTS_DIR, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.log(f"Saved: {fig_path}")
    plt.close()
    
    return data_samples


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def plot_loss_curves(results, model_type, config, exp_config):
    """
    Plot training loss curves for all complexity levels.
    model_type: 'pinn' or 'data_driven'
    """
    plt.figure(figsize=(10, 6), dpi=150)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    has_data = False
    lbfgs_start = None  # Will be computed from actual history lengths
    
    for i, K in enumerate(exp_config.K_LEVELS):
        res = results[K].get(model_type, {})
        history = res.get('history', [])
        
        if history and len(history) > 0:
            has_data = True
            label = f"K={K} ({exp_config.K_LABELS[K]})"
            plt.plot(history, label=label, linewidth=1.5, color=colors[i % len(colors)])
            
            # Determine LBFGS start from the first valid history
            # LBFGS starts after EPOCHS_ADAM iterations (Adam phase records 1 iter per epoch)
            if lbfgs_start is None:
                if model_type == 'pinn':
                    lbfgs_start = config.PINN_EPOCHS_ADAM
                else:
                    lbfgs_start = config.DD_EPOCHS_ADAM
    
    if not has_data:
        print(f"No history data found for {model_type}")
        plt.close()
        return

    # Add vertical line for LBFGS start
    if model_type == 'pinn':
        title = "PINN Training Loss"
    else:
        title = "Data-Driven Training Loss"
        
    if lbfgs_start is not None:
        plt.axvline(x=lbfgs_start, color='k', linestyle='--', alpha=0.5, linewidth=2, label='LBFGS Start')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Ensure we see the full range
    plt.xlim(left=0)
    
    plt.tight_layout()
    
    filename = f"{model_type}_loss_comparison.pdf"
    filepath = os.path.join(exp_config.PLOTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {model_type} loss comparison to {filepath}")

def save_training_histories(results, exp_config):
    """Save training histories for further visualization"""
    history_dir = os.path.join(exp_config.OUTPUT_DIR, "histories")
    os.makedirs(history_dir, exist_ok=True)
    
    for K in exp_config.K_LEVELS:
        # PINN
        pinn_hist = results[K].get('pinn', {}).get('history', [])
        if pinn_hist:
            np.save(os.path.join(history_dir, f"pinn_history_K{K}.npy"), np.array(pinn_hist))
            
        # Data-Driven
        dd_hist = results[K].get('data_driven', {}).get('history', [])
        if dd_hist:
            np.save(os.path.join(history_dir, f"dd_history_K{K}.npy"), np.array(dd_hist))
            
    print(f"Saved training histories to {history_dir}")


# ============================================================================
# MODEL TRAINING (TASK 2)
# ============================================================================

def task2_model_training(logger, exp_config):
    """
    Task 2: Train PINN and Data-Driven models for different complexity levels
    """
    logger.section("Task 2: Training and Evaluation", "=")
    
    results = {}
    config = Config()
    config.N = exp_config.N
    
    # Ensure models directory exists
    os.makedirs(exp_config.MODELS_DIR, exist_ok=True)
    
    for K in exp_config.K_LEVELS:
        complexity_label = exp_config.K_LABELS[K]
        logger.subsection(f"K={K} ({complexity_label} complexity)")
        
        config.K = K
        results[K] = {}
        
        # Generate data
        logger.log(f"\nGenerating data...")
        data_generator = Poisson_data_generator(config.N, config.K)
        force, solution = data_generator.generate()
        
        # ────────────────────────────────────────────────────────────────
        # Train or Load PINN
        # ────────────────────────────────────────────────────────────────
        pinn_path = os.path.join(exp_config.MODELS_DIR, f'pinn_K{K}.pt')
        pinn_loaded = False
        pinn_model = None
        pinn_history = []
        U_pred_pinn = None
        error_pinn = 0.0
        
        if exp_config.TRAIN_PINN:
            should_train_pinn = False
            if exp_config.RETRAIN:
                should_train_pinn = True
            elif not os.path.exists(pinn_path):
                if exp_config.RECREATE_PLOTS_ONLY:
                    logger.log(f"\nPINN model not found, skipping.")
                else:
                    should_train_pinn = True
            else:
                # Model exists and not retraining
                logger.log(f"\nLoading PINN model...")
                try:
                    pinn_model = PINN(
                        config.N_HIDDEN_LAYERS, 
                        config.WIDTH, 
                        config.N, 
                        config.DEVICE,
                        mesh=config.MESH_TYPE,
                        lambda_u=config.PINN_LAMBDA_U
                    )
                    pinn_model.load_state_dict(torch.load(pinn_path, map_location=config.DEVICE))
                    pinn_loaded = True
                    logger.log(f"PINN loaded")
                except Exception as e:
                    logger.log(f"  Failed to load PINN model: {e}")
                    if not exp_config.RECREATE_PLOTS_ONLY:
                        should_train_pinn = True
            
            if should_train_pinn:
                logger.log(f"\nTraining PINN ({config.N_HIDDEN_LAYERS} layers, width {config.WIDTH}, lambda_u={config.PINN_LAMBDA_U})")
                
                pinn_model, pinn_history = train_pinn(config, data_generator, verbose=True)
                
                logger.log(f"PINN training done: {len(pinn_history)} iterations, final loss {pinn_history[-1]:.6e}")
                
                # Save model
                torch.save(pinn_model.state_dict(), pinn_path)
                logger.log(f"Saved: {pinn_path}")
                
                # Save history
                hist_path = os.path.join(exp_config.MODELS_DIR, f'pinn_history_K{K}.npy')
                np.save(hist_path, np.array(pinn_history))
            
            # Try to load history if not trained
            if pinn_loaded and not pinn_history:
                hist_path = os.path.join(exp_config.MODELS_DIR, f'pinn_history_K{K}.npy')
                if os.path.exists(hist_path):
                    try:
                        pinn_history = np.load(hist_path).tolist()
                    except:
                        pass

            # Evaluate PINN if we have a model
            if pinn_model:
                U_pred_pinn, U_exact, error_pinn = evaluate_model(
                    pinn_model, data_generator, config, is_data_driven=False
                )
                logger.log(f"PINN test error: {error_pinn:.2f}%")
                
                # Save PINN loss curve only if trained
                if not pinn_loaded and pinn_history:
                    plt.figure(figsize=(8, 5), dpi=150)
                    plt.plot(pinn_history, linewidth=1.5, color='#2E86AB')
                    plt.xlabel('Iteration', fontsize=11)
                    plt.ylabel('Loss', fontsize=11)
                    plt.title(f'PINN Training History (K={K})', fontweight='bold', fontsize=12)
                    plt.yscale('log')
                    plt.grid(True, alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    loss_path = os.path.join(exp_config.PLOTS_DIR, f'pinn_loss_K{K}.png')
                    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                    plt.close()
        else:
            logger.log(f"\nSkipping PINN")

        results[K]['pinn'] = {
            'model': pinn_model,
            'history': pinn_history,
            'prediction': U_pred_pinn,
            'error': error_pinn,
            'final_loss': pinn_history[-1] if pinn_history else 0.0
        }
        
        # ────────────────────────────────────────────────────────────────
        # Train or Load Data-Driven
        # ────────────────────────────────────────────────────────────────
        dd_path = os.path.join(exp_config.MODELS_DIR, f'data_driven_K{K}.pt')
        dd_loaded = False
        dd_model = None
        dd_history = []
        U_pred_dd = None
        error_dd = 0.0
        
        if exp_config.TRAIN_DD:
            should_train_dd = False
            if exp_config.RETRAIN:
                should_train_dd = True
            elif not os.path.exists(dd_path):
                if exp_config.RECREATE_PLOTS_ONLY:
                    logger.log(f"\nData-Driven model not found, skipping.")
                else:
                    should_train_dd = True
            else:
                # Model exists and not retraining
                logger.log(f"\nLoading Data-Driven model...")
                try:
                    dd_model = DataDrivenModel(config.N_HIDDEN_LAYERS, config.WIDTH).to(config.DEVICE)
                    dd_model.load_state_dict(torch.load(dd_path, map_location=config.DEVICE))
                    dd_loaded = True
                    logger.log(f"Data-Driven loaded")
                except Exception as e:
                    logger.log(f"  Failed to load Data-Driven model: {e}")
                    if not exp_config.RECREATE_PLOTS_ONLY:
                        should_train_dd = True

            if should_train_dd:
                logger.log(f"\nTraining Data-Driven ({config.N_HIDDEN_LAYERS} layers, width {config.WIDTH}, batch {config.DD_BATCH_SIZE})")
                
                dd_model, dd_history = train_data_driven(config, data_generator, verbose=True)
                
                logger.log(f"Data-Driven training done: {len(dd_history)} iterations, final loss {dd_history[-1]:.6e}")
                
                # Save model
                torch.save(dd_model.state_dict(), dd_path)
                logger.log(f"Saved: {dd_path}")
                
                # Save history
                hist_path = os.path.join(exp_config.MODELS_DIR, f'dd_history_K{K}.npy')
                np.save(hist_path, np.array(dd_history))
            
            # Try to load history if not trained
            if dd_loaded and not dd_history:
                hist_path = os.path.join(exp_config.MODELS_DIR, f'dd_history_K{K}.npy')
                if os.path.exists(hist_path):
                    try:
                        dd_history = np.load(hist_path).tolist()
                    except:
                        pass
            
            # Evaluate Data-Driven if we have a model
            if dd_model:
                U_pred_dd, U_exact, error_dd = evaluate_model(
                    dd_model, data_generator, config, is_data_driven=True
                )
                logger.log(f"Data-Driven test error: {error_dd:.2f}%")
                
                # Save Data-Driven loss curve only if trained
                if not dd_loaded and dd_history:
                    plt.figure(figsize=(8, 5), dpi=150)
                    plt.plot(dd_history, linewidth=1.5, color='#A23B72')
                    plt.xlabel('Iteration', fontsize=11)
                    plt.ylabel('Loss', fontsize=11)
                    plt.title(f'Data-Driven Training History (K={K})', fontweight='bold', fontsize=12)
                    plt.yscale('log')
                    plt.grid(True, alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    loss_path = os.path.join(exp_config.PLOTS_DIR, f'data_driven_loss_K{K}.png')
                    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                    plt.close()
        else:
            logger.log(f"\nSkipping Data-Driven")

        results[K]['data_driven'] = {
            'model': dd_model,
            'history': dd_history,
            'prediction': U_pred_dd,
            'error': error_dd,
            'final_loss': dd_history[-1] if dd_history else 0.0
        }
        
        # Generate comparison plots
        logger.log(f"\nGenerating plots...")
        
        if U_pred_pinn is not None and U_pred_dd is not None:
            x = np.linspace(0, 1, config.N)
            y = np.linspace(0, 1, config.N)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # Create comprehensive comparison figure
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Row 1: PINN results
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.pcolormesh(X, Y, U_pred_pinn, shading='auto', cmap='viridis')
            ax1.set_title('PINN Prediction', fontweight='bold', fontsize=18)
            ax1.set_xlabel('x', fontsize=12)
            ax1.set_ylabel('y', fontsize=12)
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.colorbar(im1, ax=ax1)
            
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.pcolormesh(X, Y, U_exact, shading='auto', cmap='viridis')
            ax2.set_title('Exact Solution', fontweight='bold', fontsize=18)
            ax2.set_xlabel('x', fontsize=12)
            ax2.set_ylabel('y', fontsize=12)
            ax2.set_xticks([])
            ax2.set_yticks([])
            plt.colorbar(im2, ax=ax2)
            
            ax3 = fig.add_subplot(gs[0, 2])
            error_pinn_abs = np.abs(U_pred_pinn - U_exact)
            im3 = ax3.pcolormesh(X, Y, error_pinn_abs, shading='auto', cmap='Reds')
            ax3.set_title(f'PINN Error\nL2 Rel: {error_pinn:.2f}%', fontweight='bold', fontsize=18)
            ax3.set_xlabel('x', fontsize=12)
            ax3.set_ylabel('y', fontsize=12)
            ax3.set_xticks([])
            ax3.set_yticks([])
            plt.colorbar(im3, ax=ax3)
            
            # Row 2: Data-Driven results
            ax4 = fig.add_subplot(gs[1, 0])
            im4 = ax4.pcolormesh(X, Y, U_pred_dd, shading='auto', cmap='viridis')
            ax4.set_title('Data-Driven Prediction', fontweight='bold', fontsize=18)
            ax4.set_xlabel('x', fontsize=12)
            ax4.set_ylabel('y', fontsize=12)
            ax4.set_xticks([])
            ax4.set_yticks([])
            plt.colorbar(im4, ax=ax4)
            
            ax5 = fig.add_subplot(gs[1, 1])
            im5 = ax5.pcolormesh(X, Y, U_exact, shading='auto', cmap='viridis')
            ax5.set_title('Exact Solution', fontweight='bold', fontsize=18)
            ax5.set_xlabel('x', fontsize=12)
            ax5.set_ylabel('y', fontsize=12)
            ax5.set_xticks([])
            ax5.set_yticks([])
            plt.colorbar(im5, ax=ax5)
            
            ax6 = fig.add_subplot(gs[1, 2])
            error_dd_abs = np.abs(U_pred_dd - U_exact)
            im6 = ax6.pcolormesh(X, Y, error_dd_abs, shading='auto', cmap='Reds')
            ax6.set_title(f'Data-Driven Error\nL2 Rel: {error_dd:.2f}%', fontweight='bold', fontsize=18)
            ax6.set_xlabel('x', fontsize=12)
            ax6.set_ylabel('y', fontsize=12)
            ax6.set_xticks([])
            ax6.set_yticks([])
            plt.colorbar(im6, ax=ax6)
            
            fig.suptitle(f'Model Comparison - Complexity {complexity_label} (K={K})', 
                        fontsize=32, fontweight='bold', y=0.98)
            
            comp_path = os.path.join(exp_config.PLOTS_DIR, f'comparison_K{K}.png')
            plt.savefig(comp_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.log(f"Saved: {comp_path}")
    
    return results


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def generate_summary(logger, results, exp_config):
    """Generate summary plots"""
    
    config = Config()
    
    # Generate loss comparison plots
    plot_loss_curves(results, 'pinn', config, exp_config)
    plot_loss_curves(results, 'data_driven', config, exp_config)
    
    # Save histories
    save_training_histories(results, exp_config)


# ============================================================================
# SAVE RESULTS TO JSON
# ============================================================================

def save_results_json(results, exp_config):
    """Save results to JSON for further analysis"""
    
    json_results = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'N': exp_config.N,
            'K_levels': exp_config.K_LEVELS,
            'seed': exp_config.SEED
        },
        'results': {}
    }
    
    for K in exp_config.K_LEVELS:
        pinn_res = results[K].get('pinn', {})
        dd_res = results[K].get('data_driven', {})
        
        json_results['results'][K] = {
            'complexity_label': exp_config.K_LABELS[K],
            'pinn': {
                'error': float(pinn_res.get('error', 0.0)),
                'final_loss': float(pinn_res.get('final_loss', 0.0)),
                'iterations': len(pinn_res.get('history', []))
            },
            'data_driven': {
                'error': float(dd_res.get('error', 0.0)),
                'final_loss': float(dd_res.get('final_loss', 0.0)),
                'iterations': len(dd_res.get('history', []))
            }
        }
    
    with open(exp_config.RESULTS_JSON, 'w') as f:
        json.dump(json_results, f, indent=2)


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_all_experiments():
    """Execute complete experiment suite"""
    
    # Set random seed
    torch.manual_seed(ExperimentConfig.SEED)
    np.random.seed(ExperimentConfig.SEED)
    
    # Initialize logger
    logger = ResultLogger(ExperimentConfig.RESULTS_FILE)
    
    # Print header
    logger.log(f"Starting experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", to_file=False)
    logger.log(f"Results: {ExperimentConfig.RESULTS_FILE}\n", to_file=False)
    
    try:
        # Execute experiments
        start_time = datetime.now()
        
        # Task 1: Data Generation
        data_samples = task1_data_generation(logger, ExperimentConfig)
        
        # Task 2: Model Training and Evaluation
        results = task2_model_training(logger, ExperimentConfig)
        
        # Generate Summary
        generate_summary(logger, results, ExperimentConfig)
        
        # Save JSON results
        save_results_json(results, ExperimentConfig)
        logger.log(f"\nResults saved to JSON: {ExperimentConfig.RESULTS_JSON}")
        
        # Calculate total time
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.log(f"\nCompleted in {duration}", to_file=False)
        
        return results
        
    except Exception as e:
        logger.log(f"\nERROR OCCURRED: {str(e)}")
        logger.log(f"\nException details:")
        import traceback
        logger.log(traceback.format_exc())
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_all_experiments()
