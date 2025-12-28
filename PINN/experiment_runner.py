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
from task2_implementation import (
    Config, 
    Poisson_data_generator,
    train_pinn,
    train_data_driven,
    evaluate_model,
    plot_training_history,
    plot_solution_comparison
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
    K_VISUALIZATION = [1, 4, 8, 16]
    
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
        separator = char * 80
        self.log(f"\n{separator}")
        self.log(f"{title}")
        self.log(f"{separator}\n")
    
    def subsection(self, title):
        """Create a subsection header"""
        self.log(f"\n{'─'*80}")
        self.log(f"  {title}")
        self.log(f"{'─'*80}")
    
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
    logger.section("TASK 1: DATA GENERATION", "=")
    logger.log(f"Generating data for K = {exp_config.K_VISUALIZATION}")
    logger.log(f"Grid resolution: N = {exp_config.N}")
    
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
    logger.log(f"\n✓ Data visualization saved to: {fig_path}")
    plt.close()
    
    logger.log("\n✓ Task 1 completed successfully")
    
    return data_samples


# ============================================================================
# MODEL TRAINING (TASK 2)
# ============================================================================

def task2_model_training(logger, exp_config):
    """
    Task 2: Train PINN and Data-Driven models for different complexity levels
    """
    logger.section("TASK 2: MODEL TRAINING AND EVALUATION", "=")
    
    results = {}
    config = Config()
    config.N = exp_config.N
    
    for K in exp_config.K_LEVELS:
        complexity_label = exp_config.K_LABELS[K]
        logger.subsection(f"Complexity Level: {complexity_label} (K = {K})")
        
        config.K = K
        results[K] = {}
        
        # Generate data
        logger.log(f"\n[1/5] Generating data...")
        data_generator = Poisson_data_generator(config.N, config.K)
        force, solution = data_generator.generate()
        
        # ────────────────────────────────────────────────────────────────
        # Train PINN
        # ────────────────────────────────────────────────────────────────
        logger.log(f"\n[2/5] Training PINN model...")
        logger.log(f"  • Architecture: {config.N_HIDDEN_LAYERS} hidden layers, width {config.WIDTH}")
        logger.log(f"  • Adam epochs: {config.PINN_EPOCHS_ADAM}, LBFGS epochs: {config.PINN_EPOCHS_LBFGS}")
        logger.log(f"  • Lambda_u (PDE weight): {config.PINN_LAMBDA_U}")
        
        pinn_model, pinn_history = train_pinn(config, data_generator, verbose=False)
        
        logger.log(f"  ✓ PINN training completed")
        logger.log(f"  • Final loss: {pinn_history[-1]:.6e}")
        logger.log(f"  • Total iterations: {len(pinn_history)}")
        
        # Evaluate PINN
        U_pred_pinn, U_exact, error_pinn = evaluate_model(
            pinn_model, data_generator, config, is_data_driven=False
        )
        
        logger.log(f"  • L2 Relative Error: {error_pinn:.2f}%")
        
        results[K]['pinn'] = {
            'model': pinn_model,
            'history': pinn_history,
            'prediction': U_pred_pinn,
            'error': error_pinn,
            'final_loss': pinn_history[-1]
        }
        
        # Save PINN loss curve
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
        
        # ────────────────────────────────────────────────────────────────
        # Train Data-Driven
        # ────────────────────────────────────────────────────────────────
        logger.log(f"\n[3/5] Training Data-Driven model...")
        logger.log(f"  • Architecture: {config.N_HIDDEN_LAYERS} hidden layers, width {config.WIDTH}")
        logger.log(f"  • Adam epochs: {config.DD_EPOCHS_ADAM}, LBFGS epochs: {config.DD_EPOCHS_LBFGS}")
        logger.log(f"  • Batch size: {config.DD_BATCH_SIZE}")
        
        dd_model, dd_history = train_data_driven(config, data_generator, verbose=False)
        
        logger.log(f"  ✓ Data-Driven training completed")
        logger.log(f"  • Final loss: {dd_history[-1]:.6e}")
        logger.log(f"  • Total iterations: {len(dd_history)}")
        
        # Evaluate Data-Driven
        U_pred_dd, U_exact, error_dd = evaluate_model(
            dd_model, data_generator, config, is_data_driven=True
        )
        
        logger.log(f"  • L2 Relative Error: {error_dd:.2f}%")
        
        results[K]['data_driven'] = {
            'model': dd_model,
            'history': dd_history,
            'prediction': U_pred_dd,
            'error': error_dd,
            'final_loss': dd_history[-1]
        }
        
        # Save Data-Driven loss curve
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
        
        # ────────────────────────────────────────────────────────────────
        # Generate comparison plots
        # ────────────────────────────────────────────────────────────────
        logger.log(f"\n[4/5] Generating comparison plots...")
        
        x = np.linspace(0, 1, config.N)
        y = np.linspace(0, 1, config.N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Create comprehensive comparison figure
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
        
        comp_path = os.path.join(exp_config.PLOTS_DIR, f'comparison_K{K}.png')
        plt.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.log(f"  ✓ Comparison plot saved to: {comp_path}")
        
        # ────────────────────────────────────────────────────────────────
        # Save models
        # ────────────────────────────────────────────────────────────────
        logger.log(f"\n[5/5] Saving models...")
        
        os.makedirs(exp_config.MODELS_DIR, exist_ok=True)
        
        torch.save(pinn_model.state_dict(), 
                  os.path.join(exp_config.MODELS_DIR, f'pinn_K{K}.pt'))
        torch.save(dd_model.state_dict(), 
                  os.path.join(exp_config.MODELS_DIR, f'data_driven_K{K}.pt'))
        
        logger.log(f"  ✓ Models saved")
        
        logger.log(f"\n{'─'*80}")
        logger.log(f"✓ Completed K={K} ({complexity_label} complexity)")
        logger.log(f"{'─'*80}\n")
    
    return results


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def generate_summary(logger, results, exp_config):
    """Generate comprehensive results summary"""
    
    logger.section("RESULTS SUMMARY", "=")
    
    # ────────────────────────────────────────────────────────────────
    # Summary Table
    # ────────────────────────────────────────────────────────────────
    logger.subsection("Performance Comparison")
    
    headers = ["K", "Complexity", "PINN Error (%)", "DD Error (%)", "Improvement"]
    widths = [5, 12, 18, 18, 15]
    
    logger.table_header(headers, widths)
    
    for K in exp_config.K_LEVELS:
        complexity = exp_config.K_LABELS[K]
        pinn_err = results[K]['pinn']['error']
        dd_err = results[K]['data_driven']['error']
        
        # Calculate improvement (negative means PINN is better)
        improvement = ((dd_err - pinn_err) / dd_err) * 100
        improvement_str = f"{improvement:+.1f}%"
        
        logger.table_row(
            [K, complexity, f"{pinn_err:.2f}", f"{dd_err:.2f}", improvement_str],
            widths
        )
    
    # ────────────────────────────────────────────────────────────────
    # Training Statistics
    # ────────────────────────────────────────────────────────────────
    logger.subsection("Training Statistics")
    
    for K in exp_config.K_LEVELS:
        complexity = exp_config.K_LABELS[K]
        logger.log(f"\nComplexity {complexity} (K={K}):")
        
        # PINN stats
        pinn_hist = results[K]['pinn']['history']
        logger.log(f"  PINN:")
        logger.log(f"    • Total iterations: {len(pinn_hist)}")
        logger.log(f"    • Initial loss: {pinn_hist[0]:.6e}")
        logger.log(f"    • Final loss: {pinn_hist[-1]:.6e}")
        logger.log(f"    • Loss reduction: {pinn_hist[0]/pinn_hist[-1]:.2e}x")
        
        # Data-Driven stats
        dd_hist = results[K]['data_driven']['history']
        logger.log(f"  Data-Driven:")
        logger.log(f"    • Total iterations: {len(dd_hist)}")
        logger.log(f"    • Initial loss: {dd_hist[0]:.6e}")
        logger.log(f"    • Final loss: {dd_hist[-1]:.6e}")
        logger.log(f"    • Loss reduction: {dd_hist[0]/dd_hist[-1]:.2e}x")
    
    # ────────────────────────────────────────────────────────────────
    # Key Observations
    # ────────────────────────────────────────────────────────────────
    logger.subsection("Key Observations")
    
    # Find trends
    pinn_errors = [results[K]['pinn']['error'] for K in exp_config.K_LEVELS]
    dd_errors = [results[K]['data_driven']['error'] for K in exp_config.K_LEVELS]
    
    logger.log(f"1. Error Trends with Increasing Complexity:")
    logger.log(f"   • PINN error progression: {' → '.join([f'{e:.2f}%' for e in pinn_errors])}")
    logger.log(f"   • Data-Driven error progression: {' → '.join([f'{e:.2f}%' for e in dd_errors])}")
    
    pinn_increase = ((pinn_errors[-1] - pinn_errors[0]) / pinn_errors[0]) * 100
    dd_increase = ((dd_errors[-1] - dd_errors[0]) / dd_errors[0]) * 100
    
    logger.log(f"\n2. Error Growth from Low to High Complexity:")
    logger.log(f"   • PINN: {pinn_increase:+.1f}%")
    logger.log(f"   • Data-Driven: {dd_increase:+.1f}%")
    
    # Best performance
    best_k = min(exp_config.K_LEVELS, key=lambda k: results[k]['pinn']['error'])
    logger.log(f"\n3. Best PINN Performance: K={best_k} ({exp_config.K_LABELS[best_k]}) with {results[best_k]['pinn']['error']:.2f}% error")
    
    best_k_dd = min(exp_config.K_LEVELS, key=lambda k: results[k]['data_driven']['error'])
    logger.log(f"   Best Data-Driven Performance: K={best_k_dd} ({exp_config.K_LABELS[best_k_dd]}) with {results[best_k_dd]['data_driven']['error']:.2f}% error")
    
    # ────────────────────────────────────────────────────────────────
    # Create summary plots
    # ────────────────────────────────────────────────────────────────
    logger.subsection("Generating Summary Visualizations")
    
    # Error comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Error vs Complexity
    K_vals = exp_config.K_LEVELS
    ax1.plot(K_vals, pinn_errors, 'o-', linewidth=2, markersize=8, 
             label='PINN', color='#2E86AB')
    ax1.plot(K_vals, dd_errors, 's-', linewidth=2, markersize=8, 
             label='Data-Driven', color='#A23B72')
    ax1.set_xlabel('K (Frequency Modes)', fontsize=11)
    ax1.set_ylabel('L2 Relative Error (%)', fontsize=11)
    ax1.set_title('Error vs Problem Complexity', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(K_vals)
    
    # Plot 2: Training convergence comparison
    for K in exp_config.K_LEVELS:
        label = exp_config.K_LABELS[K]
        pinn_hist = results[K]['pinn']['history']
        # Subsample for clarity
        step = max(1, len(pinn_hist) // 1000)
        ax2.plot(pinn_hist[::step], label=f'K={K} ({label})', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('PINN Loss', fontsize=11)
    ax2.set_title('PINN Training Convergence', fontweight='bold', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    summary_path = os.path.join(exp_config.PLOTS_DIR, 'summary_comparison.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.log(f"  ✓ Summary plots saved to: {summary_path}")


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
        json_results['results'][K] = {
            'complexity_label': exp_config.K_LABELS[K],
            'pinn': {
                'error': float(results[K]['pinn']['error']),
                'final_loss': float(results[K]['pinn']['final_loss']),
                'iterations': len(results[K]['pinn']['history'])
            },
            'data_driven': {
                'error': float(results[K]['data_driven']['error']),
                'final_loss': float(results[K]['data_driven']['final_loss']),
                'iterations': len(results[K]['data_driven']['history'])
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
    logger.log("="*80, to_file=False)
    logger.log("AISE 2026 - FINAL PROJECT - PROBLEM 1", to_file=False)
    logger.log("Visualizing Loss Landscapes: PINNs vs. Data-Driven", to_file=False)
    logger.log("="*80, to_file=False)
    logger.log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", to_file=False)
    logger.log(f"Results will be saved to: {ExperimentConfig.RESULTS_FILE}", to_file=False)
    logger.log("="*80 + "\n", to_file=False)
    
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
        logger.log(f"\n✓ Results saved to JSON: {ExperimentConfig.RESULTS_JSON}")
        
        # Calculate total time
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Final summary
        logger.section("EXPERIMENT COMPLETED SUCCESSFULLY", "=")
        logger.log(f"Total execution time: {duration}")
        logger.log(f"\nAll results have been saved to: {ExperimentConfig.OUTPUT_DIR}/")
        logger.log(f"  • Detailed results: {ExperimentConfig.RESULTS_FILE}")
        logger.log(f"  • JSON results: {ExperimentConfig.RESULTS_JSON}")
        logger.log(f"  • Plots: {ExperimentConfig.PLOTS_DIR}/")
        logger.log(f"  • Models: {ExperimentConfig.MODELS_DIR}/")
        
        logger.log("\n" + "="*80)
        logger.log("Thank you for running the AISE 2026 experiments!")
        logger.log("="*80)
        
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
    print("\nStarting AISE 2026 Final Project Experiments...")
    print("This will take several minutes to complete.\n")
    
    results = run_all_experiments()
    
    print("\nAll experiments completed successfully!")
    print(f"Check '{ExperimentConfig.RESULTS_FILE}' for detailed results")
    print(f"Check '{ExperimentConfig.PLOTS_DIR}/' for all visualizations\n")