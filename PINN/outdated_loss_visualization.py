import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from datetime import datetime
from pathlib import Path
import json

# Import from refactored code
from task2_implementation import (
    Config,
    Poisson_data_generator,
    train_pinn,
    train_data_driven,
    PINN,
    DataDrivenModel
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class LandscapeConfig:
    """Configuration for loss landscape visualization"""
    
    # Complexity levels to visualize
    K_LEVELS = [16] # [1, 4, 16] # Low, Medium, High
    K_LABELS = {1: "Low", 4: "Medium", 16: "High"}
    
    # Grid resolution
    N = 64
    
    # Landscape sampling parameters
    ALPHA_RANGE = (-0.5, 0.5)  # Range for first direction
    BETA_RANGE = (-0.5, 0.5)   # Range for second direction
    N_POINTS = 30              # Grid resolution (150x150 = 22500 evaluations)
    
    # Direction generation method
    DIRECTION_METHOD = "gradient"  # "random", "filter_normalized", "gradient"
    
    # Output directories
    OUTPUT_DIR = "results/loss_landscapes_residual"
    PLOTS_2D_DIR = "results/loss_landscapes_residual/2d_contours"
    PLOTS_3D_DIR = "results/loss_landscapes_residual/3d_surfaces"
    DATA_DIR = "results/loss_landscapes_residual/data"
    
    # Results file
    RESULTS_FILE = "results/loss_landscapes_residual/landscape_analysis.txt"
    
    # Random seed
    SEED = 42


# ============================================================================
# PARAMETER MANIPULATION UTILITIES
# ============================================================================

def get_parameters_as_vector(model):
    """
    Extract all model parameters and flatten them into a single 1D vector
    
    Returns:
        torch.Tensor: Flattened parameter vector
    """
    parameters = []
    for param in model.parameters():
        parameters.append(param.data.view(-1))
    return torch.cat(parameters)


def set_parameters_from_vector(model, vector):
    """
    Set model parameters from a flat vector
    
    Parameters:
        model: Neural network model
        vector: Flattened parameter vector
    """
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = vector[pointer:pointer + num_param].view_as(param).clone()
        pointer += num_param


def get_num_parameters(model):
    """Count total number of parameters in model"""
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# DIRECTION VECTOR GENERATION
# ============================================================================

def generate_random_directions(model):
    """
    Generate two random direction vectors (normalized)
    
    Simple approach: random Gaussian directions
    """
    theta_star = get_parameters_as_vector(model)
    
    # Random Gaussian directions
    delta = torch.randn_like(theta_star)
    eta = torch.randn_like(theta_star)
    
    # Normalize to unit length
    delta = delta / torch.norm(delta)
    eta = eta / torch.norm(eta)
    
    return delta, eta


def generate_filter_normalized_directions(model):
    """
    Generate direction vectors normalized by filter norms
    
    This method (from Li et al. 2018) accounts for different scales 
    of parameters across layers, giving more meaningful visualizations.
    """
    delta = []
    eta = []
    
    for param in model.parameters():
        # Random directions
        d = torch.randn_like(param)
        e = torch.randn_like(param)
        
        # Normalize by the norm of the parameter
        # This ensures directions scale appropriately with parameter magnitude
        param_norm = torch.norm(param)
        if param_norm > 1e-10:
            d = d / torch.norm(d) * param_norm
            e = e / torch.norm(e) * param_norm
        
        delta.append(d.view(-1))
        eta.append(e.view(-1))
    
    delta = torch.cat(delta)
    eta = torch.cat(eta)
    
    # Final normalization
    delta = delta / torch.norm(delta)
    eta = eta / torch.norm(eta)
    
    return delta, eta

def generate_gradient_directions(model, loss_computer, device):
    """
    Generates directions based on the local gradient.
    This guarantees we slice through the steepest, roughest part of the landscape.
    """
    print("   -> Computing gradients for direction generation...")
    
    # 1. Compute Gradient (Direction of Steepest Ascent)
    model.eval()
    model.zero_grad()
    
    # We must enable grad tracking just for this calculation
    # (Even if model is in eval mode, we need dLoss/dParams)
    params_list = list(model.parameters())
    
    # Compute loss
    loss = loss_computer()
    loss.backward()
    
    # Collect gradient into a single vector
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).clone())
        else:
            grads.append(torch.zeros_like(param.data.view(-1)))
    
    # Create Delta (Gradient Direction)
    delta = torch.cat(grads)
    
    # Check if gradient is effectively zero (perfect minimum)
    grad_norm = torch.norm(delta)
    if grad_norm < 1e-7:
        print("   -> Warning: Gradient is near zero. Using random direction instead.")
        delta = torch.randn_like(delta)
    
    delta = delta / (torch.norm(delta) + 1e-10) # Normalize
    
    # 2. Create Eta (Random Orthogonal Direction)
    eta = torch.randn_like(delta)
    
    # Gram-Schmidt orthogonalization: eta = eta - proj_delta(eta)
    projection = torch.dot(eta, delta) * delta
    eta = eta - projection
    eta = eta / (torch.norm(eta) + 1e-10) # Normalize
    
    return delta, eta


def generate_directions(model, method="filter_normalized", loss_computer=None, device='cpu'):
    """
    Generate direction vectors using specified method
    
    Parameters:
        model: Neural network model
        method: "random", "filter_normalized", or "gradient"
        loss_computer: Function to compute loss (required for "gradient")
        device: torch device (required for "gradient")
    
    Returns:
        delta, eta: Two direction vectors
    """
    if method == "random":
        return generate_random_directions(model)
    elif method == "filter_normalized":
        return generate_filter_normalized_directions(model)
    elif method == "gradient":
        if loss_computer is None:
            raise ValueError("loss_computer is required for gradient direction method")
        return generate_gradient_directions(model, loss_computer, device)
    else:
        raise ValueError(f"Unknown direction method: {method}")


# ============================================================================
# LOSS COMPUTATION
# ============================================================================

def create_pinn_loss_computer(pinn_model, data_generator, config, only_residual=True):
    """
    Create loss computation function for PINN
    MODIFIED: Returns only the PDE residual to visualize the roughness.
    """
    # Get boundary and interior data
    inp_b, u_b = next(iter(pinn_model.training_set_b))
    inp_int, _ = next(iter(pinn_model.training_set_int))
    
    # Forcing term wrapper
    def forcing_term(x):
        x_np = x[:, 0].detach().cpu().numpy()
        y_np = x[:, 1].detach().cpu().numpy()
        f_np = data_generator.forcing_term(x_np, y_np)
        return torch.tensor(f_np, dtype=torch.float32, device=x.device).reshape(-1)
    
    def compute_loss():
        # Re-implement logic here to isolate components
        # 1. Boundary Loss (We ignore this now)
        u_pred_b = pinn_model.apply_boundary_conditions(inp_b)
        r_b = u_pred_b.reshape(-1) - u_b.reshape(-1)
        loss_b = torch.mean(r_b ** 2)

        # 2. PDE Residual (The "Rough" part)
        r_int = pinn_model.compute_pde_residual(inp_int, forcing_term)
        loss_int = torch.mean(r_int ** 2)
        
        if only_residual:
            # THIS IS THE KEY CHANGE
            return loss_int
        else:
            return loss_b * pinn_model.lambda_u + loss_int
    
    return compute_loss


def create_data_driven_loss_computer(dd_model, coords, targets):
    """
    Create loss computation function for Data-Driven model
    
    Returns:
        Function that computes supervised loss for current model state
    """
    def compute_loss():
        """Compute supervised learning loss"""
        pred = dd_model(coords)
        return torch.mean((pred.reshape(-1) - targets.reshape(-1)) ** 2)
    
    return compute_loss


def compute_loss_at_point(model, theta_star, delta, eta, alpha, beta, 
                          loss_computer, device):
    """
    Compute loss at perturbed parameters: θ* + α·δ + β·η
    
    Parameters:
        model: Neural network model
        theta_star: Converged parameters
        delta, eta: Direction vectors
        alpha, beta: Scalar coefficients
        loss_computer: Function to compute loss
        device: torch device
    
    Returns:
        float: Loss value at perturbed point
    """
    # Perturb parameters
    theta_perturbed = theta_star + alpha * delta + beta * eta
    
    # Set model to perturbed parameters
    set_parameters_from_vector(model, theta_perturbed)
    
    # Compute loss
    model.eval()
    
    # We cannot use torch.no_grad() here because PINN loss computation 
    # requires gradients with respect to input coordinates (for PDE residual)
    # Even though we don't need gradients w.r.t parameters, the graph must be built
    # for the inputs.
    loss = loss_computer()
    
    return loss.item()


# ============================================================================
# LANDSCAPE COMPUTATION
# ============================================================================

def compute_loss_landscape(model, theta_star, delta, eta, loss_computer,
                          alpha_range, beta_range, n_points=25, device='cpu'):
    """
    Compute loss landscape over a 2D grid
    
    Parameters:
        model: Neural network model
        theta_star: Converged parameters
        delta, eta: Direction vectors
        loss_computer: Function to compute loss
        alpha_range: tuple (min, max) for alpha direction
        beta_range: tuple (min, max) for beta direction
        n_points: Grid resolution (n_points x n_points)
        device: torch device
    
    Returns:
        alphas: array of alpha values
        betas: array of beta values
        loss_surface: 2D array of loss values
    """
    # Create grid
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    
    # Initialize loss surface
    loss_surface = np.zeros((n_points, n_points))
    
    print(f"  Computing {n_points}x{n_points} grid ({n_points**2} evaluations)...")
    
    # Compute loss at each grid point
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            loss = compute_loss_at_point(
                model, theta_star, delta, eta, 
                alpha, beta, loss_computer, device
            )
            loss_surface[i, j] = loss
        
        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"    Progress: {i+1}/{n_points} rows completed")
    
    # Restore original parameters
    set_parameters_from_vector(model, theta_star)
    
    return alphas, betas, loss_surface


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_2d_contour(alphas, betas, loss_surface, title, filename, 
                   center_loss=None):
    """
    Create 2D contour plot of loss landscape
    
    Parameters:
        alphas, betas: Grid coordinates
        loss_surface: 2D loss values
        title: Plot title
        filename: Save path
        center_loss: Loss at θ* (for reference)
    """
    plt.figure(figsize=(10, 8))
    
    # Use log scale for better visualization
    loss_min = loss_surface.min()
    loss_max = loss_surface.max()
    
    # Create logarithmic levels
    if loss_min > 0:
        levels = np.logspace(
            np.log10(loss_min), 
            np.log10(loss_max), 
            30
        )
    else:
        levels = 30
    
    # Contour filled plot
    contour = plt.contourf(alphas, betas, loss_surface.T, 
                          levels=levels, cmap='viridis')
    cbar = plt.colorbar(contour, label='Loss')
    
    # Add contour lines
    plt.contour(alphas, betas, loss_surface.T, 
               levels=levels, colors='white', 
               alpha=0.3, linewidths=0.5)
    
    # Mark the optimum (center at alpha=0, beta=0)
    center_idx_alpha = len(alphas) // 2
    center_idx_beta = len(betas) // 2
    center_actual_loss = loss_surface[center_idx_alpha, center_idx_beta]
    
    plt.plot(0, 0, 'r*', markersize=20, label=f'θ* (Loss: {center_actual_loss:.2e})')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.xlabel('$\\alpha$ (Direction $\\delta$)', fontsize=13)
    plt.ylabel('$\\beta$ (Direction $\\eta$)', fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save as PNG and PDF
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()


def plot_3d_surface(alphas, betas, loss_surface, title, filename):
    """
    Create 3D surface plot of loss landscape
    
    Parameters:
        alphas, betas: Grid coordinates
        loss_surface: 2D loss values
        title: Plot title
        filename: Save path
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    Alpha, Beta = np.meshgrid(alphas, betas)
    
    # Plot surface with transparency
    surf = ax.plot_surface(Alpha, Beta, loss_surface.T, 
                          cmap='viridis', alpha=0.85,
                          edgecolor='none', antialiased=True)
    
    # Mark optimum
    center_idx_alpha = len(alphas) // 2
    center_idx_beta = len(betas) // 2
    center_loss = loss_surface[center_idx_alpha, center_idx_beta]
    
    ax.scatter([0], [0], [center_loss], 
              color='red', s=150, marker='*', 
              label=f'θ* (Loss: {center_loss:.2e})', zorder=10)
    
    # Labels and title
    ax.set_xlabel('$\\alpha$ (Direction $\\delta$)', fontsize=11, labelpad=10)
    ax.set_ylabel('$\\beta$ (Direction $\\eta$)', fontsize=11, labelpad=10)
    ax.set_zlabel('Loss', fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Legend
    ax.legend(fontsize=9)
    
    # View angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    
    # Plot saved, closing to allow continuous execution
    plt.close()


def create_comparison_plot(landscapes_data, K, output_dir):
    """
    Create side-by-side comparison of PINN vs Data-Driven landscapes
    
    Parameters:
        landscapes_data: dict with 'pinn' and 'data_driven' landscape data
        K: Complexity level
        output_dir: Directory to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    model_types = ['pinn', 'data_driven']
    titles = ['PINN Loss Landscape', 'Data-Driven Loss Landscape']
    
    # Determine common loss range for fair comparison
    all_losses = []
    for model_type in model_types:
        all_losses.append(landscapes_data[model_type]['loss_surface'].flatten())
    all_losses = np.concatenate(all_losses)
    
    loss_min = all_losses.min()
    loss_max = all_losses.max()
    
    if loss_min > 0:
        levels = np.logspace(np.log10(loss_min), np.log10(loss_max), 30)
    else:
        levels = 30
    
    for idx, (model_type, title) in enumerate(zip(model_types, titles)):
        alphas = landscapes_data[model_type]['alphas']
        betas = landscapes_data[model_type]['betas']
        loss_surface = landscapes_data[model_type]['loss_surface']
        
        # Contour plot with common levels
        contour = axes[idx].contourf(alphas, betas, loss_surface.T, 
                                     levels=levels, cmap='viridis')
        plt.colorbar(contour, ax=axes[idx], label='Loss')
        
        axes[idx].contour(alphas, betas, loss_surface.T, 
                         levels=levels, colors='white', 
                         alpha=0.3, linewidths=0.5)
        
        # Mark optimum
        axes[idx].plot(0, 0, 'r*', markersize=20, label='θ*')
        
        axes[idx].set_xlabel('$\\alpha$ (Direction $\\delta$)', fontsize=12)
        axes[idx].set_ylabel('$\\beta$ (Direction $\\eta$)', fontsize=12)
        axes[idx].set_title(title, fontsize=13, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(f'Loss Landscape Comparison (K={K})', 
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'comparison_K{K}.png')
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    return filename


# ============================================================================
# LANDSCAPE ANALYSIS
# ============================================================================

def analyze_landscape(alphas, betas, loss_surface, model_type, K):
    """
    Compute quantitative metrics for landscape characterization
    
    Returns:
        dict: Analysis metrics
    """
    # Center loss (at θ*)
    center_idx = len(alphas) // 2
    center_loss = loss_surface[center_idx, center_idx]
    
    # Min and max loss
    min_loss = loss_surface.min()
    max_loss = loss_surface.max()
    
    # Loss range
    loss_range = max_loss - min_loss
    
    # Compute gradient magnitudes (roughness indicator)
    grad_y, grad_x = np.gradient(loss_surface)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mean_gradient = np.mean(gradient_magnitude)
    max_gradient = np.max(gradient_magnitude)
    
    # Compute curvature (via Laplacian - sharpness indicator)
    laplacian = np.abs(np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1))
    mean_curvature = np.mean(laplacian)
    max_curvature = np.max(laplacian)
    
    # Count local minima (approximate)
    # A point is a local minimum if it's lower than all 8 neighbors
    local_minima_count = 0
    for i in range(1, loss_surface.shape[0] - 1):
        for j in range(1, loss_surface.shape[1] - 1):
            neighbors = loss_surface[i-1:i+2, j-1:j+2]
            if loss_surface[i, j] == neighbors.min():
                local_minima_count += 1
    
    # Sharpness: how much loss increases when moving away from optimum
    # Sample points at fixed radius from center
    radius = 0.5  # in normalized coordinates
    n_samples = 8
    angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    alpha_center = alphas[center_idx]
    beta_center = betas[center_idx]
    
    losses_at_radius = []
    for angle in angles:
        alpha_offset = radius * np.cos(angle)
        beta_offset = radius * np.sin(angle)
        
        # Find nearest grid point
        alpha_idx = np.argmin(np.abs(alphas - (alpha_center + alpha_offset)))
        beta_idx = np.argmin(np.abs(betas - (beta_center + beta_offset)))
        
        if 0 <= alpha_idx < len(alphas) and 0 <= beta_idx < len(betas):
            losses_at_radius.append(loss_surface[alpha_idx, beta_idx])
    
    if losses_at_radius:
        sharpness = np.mean(losses_at_radius) - center_loss
    else:
        sharpness = 0
    
    return {
        'model_type': model_type,
        'K': K,
        'center_loss': center_loss,
        'min_loss': min_loss,
        'max_loss': max_loss,
        'loss_range': loss_range,
        'mean_gradient': mean_gradient,
        'max_gradient': max_gradient,
        'mean_curvature': mean_curvature,
        'max_curvature': max_curvature,
        'local_minima_count': local_minima_count,
        'sharpness': sharpness
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def visualize_single_landscape(model, model_type, K, loss_computer, 
                              config, landscape_config):
    """
    Complete pipeline for visualizing a single model's loss landscape
    
    Parameters:
        model: Trained neural network
        model_type: "pinn" or "data_driven"
        K: Complexity level
        loss_computer: Function to compute loss
        config: Model configuration
        landscape_config: Landscape visualization configuration
    
    Returns:
        dict: Landscape data and analysis
    """
    print(f"\n{'-'*70}")
    print(f"Visualizing {model_type.upper()} landscape (K={K})")
    print(f"{'-'*70}")
    
    # Step 1: Get converged parameters
    print("[1/5] Extracting parameters...")
    theta_star = get_parameters_as_vector(model)
    n_params = theta_star.shape[0]
    print(f"  Total parameters: {n_params:,}")
    
    # Step 2: Generate directions
    print(f"[2/5] Generating direction vectors ({landscape_config.DIRECTION_METHOD})...")
    delta, eta = generate_directions(model, landscape_config.DIRECTION_METHOD, loss_computer, config.DEVICE)
    print(f"  Direction delta norm: {torch.norm(delta).item():.6f}")
    print(f"  Direction eta norm: {torch.norm(eta).item():.6f}")
    print(f"  Orthogonality (delta*eta): {torch.dot(delta, eta).item():.6f}")
    
    # Step 3: Compute landscape
    print("[3/5] Computing loss landscape...")
    alphas, betas, loss_surface = compute_loss_landscape(
        model=model,
        theta_star=theta_star,
        delta=delta,
        eta=eta,
        loss_computer=loss_computer,
        alpha_range=landscape_config.ALPHA_RANGE,
        beta_range=landscape_config.BETA_RANGE,
        n_points=landscape_config.N_POINTS,
        device=config.DEVICE
    )
    
    print(f"  Loss at theta*: {loss_surface[landscape_config.N_POINTS//2, landscape_config.N_POINTS//2]:.6e}")
    print(f"  Min loss: {loss_surface.min():.6e}")
    print(f"  Max loss: {loss_surface.max():.6e}")
    
    # Step 4: Analyze landscape
    print("[4/5] Analyzing landscape characteristics...")
    analysis = analyze_landscape(alphas, betas, loss_surface, model_type, K)
    
    print(f"  Mean gradient magnitude: {analysis['mean_gradient']:.6e}")
    print(f"  Mean curvature: {analysis['mean_curvature']:.6e}")
    print(f"  Approximate local minima: {analysis['local_minima_count']}")
    print(f"  Sharpness metric: {analysis['sharpness']:.6e}")
    
    # Step 5: Create visualizations
    print("[5/5] Creating visualizations...")
    
    # 2D contour
    filename_2d = os.path.join(
        landscape_config.PLOTS_2D_DIR,
        f'landscape_2d_{model_type}_K{K}.png'
    )
    plot_2d_contour(
        alphas, betas, loss_surface,
        title=f'{model_type.upper()} Loss Landscape (K={K})',
        filename=filename_2d,
        center_loss=analysis['center_loss']
    )
    print(f"  2D contour saved: {filename_2d}")
    
    # 3D surface
    filename_3d = os.path.join(
        landscape_config.PLOTS_3D_DIR,
        f'landscape_3d_{model_type}_K{K}.png'
    )
    plot_3d_surface(
        alphas, betas, loss_surface,
        title=f'{model_type.upper()} Loss Landscape (K={K})',
        filename=filename_3d
    )
    print(f"  3D surface saved: {filename_3d}")
    
    # Save landscape data
    data_file = os.path.join(
        landscape_config.DATA_DIR,
        f'landscape_{model_type}_K{K}.npz'
    )
    np.savez(data_file, 
             alphas=alphas, 
             betas=betas, 
             loss_surface=loss_surface,
             analysis=analysis)
    print(f"  Data saved: {data_file}")
    
    print(f"{model_type.upper()} landscape complete\n")
    
    return {
        'alphas': alphas,
        'betas': betas,
        'loss_surface': loss_surface,
        'analysis': analysis
    }


def run_landscape_visualization_suite():
    """
    Execute complete loss landscape visualization suite
    """
    # Initialize
    torch.manual_seed(LandscapeConfig.SEED)
    np.random.seed(LandscapeConfig.SEED)
    
    # Create output directories
    for directory in [LandscapeConfig.OUTPUT_DIR,
                     LandscapeConfig.PLOTS_2D_DIR,
                     LandscapeConfig.PLOTS_3D_DIR,
                     LandscapeConfig.DATA_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize results file
    results_file = open(LandscapeConfig.RESULTS_FILE, 'w', encoding='utf-8')
    
    def log(message):
        """Log to both console and file"""
        print(message)
        results_file.write(message + '\n')
    
    # Header
    log("="*80)
    log("AISE 2026 - Loss Landscape Visualization (Task 3 - BONUS)")
    log("="*80)
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Configuration:")
    log(f"  • Grid resolution: {LandscapeConfig.N}")
    log(f"  • Landscape grid: {LandscapeConfig.N_POINTS}x{LandscapeConfig.N_POINTS}")
    log(f"  • alpha range: {LandscapeConfig.ALPHA_RANGE}")
    log(f"  • beta range: {LandscapeConfig.BETA_RANGE}")
    log(f"  • Direction method: {LandscapeConfig.DIRECTION_METHOD}")
    log("="*80 + "\n")
    
    # Store all results
    all_landscapes = {}
    all_analyses = {}
    
    # Main configuration
    config = Config()
    config.N = LandscapeConfig.N
    
    # Process each complexity level
    for K in LandscapeConfig.K_LEVELS:
        complexity_label = LandscapeConfig.K_LABELS[K]
        log(f"\n{'='*80}")
        log(f"COMPLEXITY LEVEL: {complexity_label} (K={K})")
        log(f"{'='*80}")
        
        config.K = K
        all_landscapes[K] = {}
        all_analyses[K] = {}
        
        # Generate data
        log("\nGenerating problem data...")
        data_generator = Poisson_data_generator(config.N, config.K)
        force, solution = data_generator.generate()
        log("  Data generated")
        
        # ────────────────────────────────────────────────────────────────
        # Load models
        # ────────────────────────────────────────────────────────────────
        log("\n" + "─"*70)
        log("LOADING MODELS")
        log("─"*70)
        
        # Load PINN
        log("\nLoading PINN...")
        pinn_path = os.path.join("results/models", f"pinn_K{K}.pt")
        if os.path.exists(pinn_path):
            pinn_model = PINN(
                config.N_HIDDEN_LAYERS, 
                config.WIDTH, 
                config.N, 
                config.DEVICE,
                mesh=config.MESH_TYPE,
                lambda_u=config.PINN_LAMBDA_U
            )
            pinn_model.load_state_dict(torch.load(pinn_path, map_location=config.DEVICE))
            log(f"  PINN loaded from {pinn_path}")
        else:
            log(f"  PINN model not found at {pinn_path}. Training new model...")
            pinn_model, pinn_history = train_pinn(config, data_generator, verbose=False)
            log(f"  PINN trained - Final loss: {pinn_history[-1]:.6e}")
        
        # Load Data-Driven
        log("\nLoading Data-Driven model...")
        dd_path = os.path.join("results/models", f"data_driven_K{K}.pt")
        if os.path.exists(dd_path):
            dd_model = DataDrivenModel(config.N_HIDDEN_LAYERS, config.WIDTH).to(config.DEVICE)
            dd_model.load_state_dict(torch.load(dd_path, map_location=config.DEVICE))
            log(f"  Data-Driven model loaded from {dd_path}")
        else:
            log(f"  Data-Driven model not found at {dd_path}. Training new model...")
            dd_model, dd_history = train_data_driven(config, data_generator, verbose=False)
            log(f"  Data-Driven trained - Final loss: {dd_history[-1]:.6e}")
        
        # ────────────────────────────────────────────────────────────────
        # Prepare loss computers
        # ────────────────────────────────────────────────────────────────
        log("\n" + "─"*70)
        log("PREPARING LOSS FUNCTIONS")
        log("─"*70)
        
        # PINN loss computer
        pinn_loss_fn = create_pinn_loss_computer(pinn_model, data_generator, config)
        
        # Data-Driven loss computer
        x = np.linspace(0, 1, config.N)
        y = np.linspace(0, 1, config.N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        coords = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32,
            device=config.DEVICE
        )
        targets = torch.tensor(
            solution.flatten() * config.DD_SCALE_FACTOR,
            dtype=torch.float32,
            device=config.DEVICE
        )
        dd_loss_fn = create_data_driven_loss_computer(dd_model, coords, targets)
        
        log("  Loss functions prepared")
        
        # ────────────────────────────────────────────────────────────────
        # Visualize PINN landscape
        # ────────────────────────────────────────────────────────────────
        log("\n" + "─"*70)
        log("PINN LOSS LANDSCAPE")
        log("─"*70)
        
        pinn_landscape = visualize_single_landscape(
            pinn_model, 'pinn', K, pinn_loss_fn,
            config, LandscapeConfig
        )
        all_landscapes[K]['pinn'] = pinn_landscape
        all_analyses[K]['pinn'] = pinn_landscape['analysis']
        
        # ────────────────────────────────────────────────────────────────
        # Visualize Data-Driven landscape
        # ────────────────────────────────────────────────────────────────
        log("\n" + "─"*70)
        log("DATA-DRIVEN LOSS LANDSCAPE")
        log("─"*70)
        
        dd_landscape = visualize_single_landscape(
            dd_model, 'data_driven', K, dd_loss_fn,
            config, LandscapeConfig
        )
        all_landscapes[K]['data_driven'] = dd_landscape
        all_analyses[K]['data_driven'] = dd_landscape['analysis']
        
        # ────────────────────────────────────────────────────────────────
        # Create comparison plot
        # ────────────────────────────────────────────────────────────────
        log("\n" + "─"*70)
        log("CREATING COMPARISON PLOT")
        log("─"*70)
        
        comp_file = create_comparison_plot(
            all_landscapes[K], K, LandscapeConfig.OUTPUT_DIR
        )
        log(f"  Comparison saved: {comp_file}")
    
    # ────────────────────────────────────────────────────────────────
    # Generate Summary Analysis
    # ────────────────────────────────────────────────────────────────
    log("\n" + "="*80)
    log("SUMMARY ANALYSIS")
    log("="*80)
    
    # Create summary table
    log("\n" + "─"*80)
    log("LANDSCAPE CHARACTERISTICS COMPARISON")
    log("─"*80)
    
    header = f"{'K':<5} | {'Model':<12} | {'Sharpness':<12} | {'Mean Grad':<12} | {'Mean Curv':<12} | {'Local Min':<10}"
    log(header)
    log("─"*80)
    
    for K in LandscapeConfig.K_LEVELS:
        for model_type in ['pinn', 'data_driven']:
            analysis = all_analyses[K][model_type]
            row = (f"{K:<5} | {model_type:<12} | "
                  f"{analysis['sharpness']:<12.4e} | "
                  f"{analysis['mean_gradient']:<12.4e} | "
                  f"{analysis['mean_curvature']:<12.4e} | "
                  f"{analysis['local_minima_count']:<10}")
            log(row)
    
    # Key observations
    log("\n" + "─"*80)
    log("KEY OBSERVATIONS")
    log("─"*80)
    
    log("\n1. Sharpness Comparison:")
    for K in LandscapeConfig.K_LEVELS:
        pinn_sharp = all_analyses[K]['pinn']['sharpness']
        dd_sharp = all_analyses[K]['data_driven']['sharpness']
        ratio = pinn_sharp / dd_sharp if dd_sharp > 0 else float('inf')
        log(f"   K={K}: PINN/DD sharpness ratio = {ratio:.2f}x")
    
    log("\n2. Gradient Magnitude (Roughness):")
    for K in LandscapeConfig.K_LEVELS:
        pinn_grad = all_analyses[K]['pinn']['mean_gradient']
        dd_grad = all_analyses[K]['data_driven']['mean_gradient']
        ratio = pinn_grad / dd_grad if dd_grad > 0 else float('inf')
        log(f"   K={K}: PINN/DD gradient ratio = {ratio:.2f}x")
    
    log("\n3. Complexity Impact:")
    pinn_sharps = [all_analyses[K]['pinn']['sharpness'] for K in LandscapeConfig.K_LEVELS]
    dd_sharps = [all_analyses[K]['data_driven']['sharpness'] for K in LandscapeConfig.K_LEVELS]
    
    if len(pinn_sharps) > 1:
        pinn_growth = ((pinn_sharps[-1] - pinn_sharps[0]) / abs(pinn_sharps[0])) * 100
        dd_growth = ((dd_sharps[-1] - dd_sharps[0]) / abs(dd_sharps[0])) * 100
        log(f"   PINN sharpness growth (K=1 to K={LandscapeConfig.K_LEVELS[-1]}): {pinn_growth:+.1f}%")
        log(f"   DD sharpness growth (K=1 to K={LandscapeConfig.K_LEVELS[-1]}): {dd_growth:+.1f}%")
    
    # ────────────────────────────────────────────────────────────────
    # Save summary JSON
    # ────────────────────────────────────────────────────────────────
    summary_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'N': LandscapeConfig.N,
            'n_points': LandscapeConfig.N_POINTS,
            'alpha_range': LandscapeConfig.ALPHA_RANGE,
            'beta_range': LandscapeConfig.BETA_RANGE,
            'direction_method': LandscapeConfig.DIRECTION_METHOD
        },
        'analyses': {}
    }
    
    for K in LandscapeConfig.K_LEVELS:
        summary_data['analyses'][K] = {
            'pinn': all_analyses[K]['pinn'],
            'data_driven': all_analyses[K]['data_driven']
        }
    
    json_file = os.path.join(LandscapeConfig.OUTPUT_DIR, 'landscape_summary.json')
    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    log(f"\n  Summary JSON saved: {json_file}")
    
    # ────────────────────────────────────────────────────────────────
    # Final summary
    # ────────────────────────────────────────────────────────────────
    log("\n" + "="*80)
    log("VISUALIZATION COMPLETE")
    log("="*80)
    log(f"\nAll results saved to: {LandscapeConfig.OUTPUT_DIR}/")
    log(f"   2D contours: {LandscapeConfig.PLOTS_2D_DIR}/")
    log(f"   3D surfaces: {LandscapeConfig.PLOTS_3D_DIR}/")
    log(f"   Raw data: {LandscapeConfig.DATA_DIR}/")
    log(f"   Analysis: {LandscapeConfig.RESULTS_FILE}")
    log(f"   Summary JSON: {json_file}")
    
    log("\n" + "="*80)
    log("Thank you for running the loss landscape analysis!")
    log("="*80)
    
    results_file.close()
    
    return all_landscapes, all_analyses


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\nStarting Loss Landscape Visualization...")
    
    landscapes, analyses = run_landscape_visualization_suite()
    
    print("\nLoss landscape visualization complete!")