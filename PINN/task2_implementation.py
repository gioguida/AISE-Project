import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from task1_data_generation import Poisson_data_generator

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for experiments"""
    
    # Grid parameters
    N = 64  # Spatial resolution
    
    # Model architecture
    N_HIDDEN_LAYERS = 4
    WIDTH = 128
    
    # Problem complexity
    K = 4  # Number of frequency modes (1, 4, 8, 16)
    
    # Training parameters - PINN
    PINN_EPOCHS_ADAM = 4000
    PINN_EPOCHS_LBFGS = 100
    PINN_LR_ADAM = 0.001
    PINN_LR_LBFGS = 0.5
    PINN_LAMBDA_U = 100.0  # Weight for PDE residual loss
    
    # Training parameters - Data-Driven
    DD_EPOCHS_ADAM = 4000
    DD_EPOCHS_LBFGS = 100
    DD_LR_ADAM = 0.001
    DD_LR_LBFGS = 0.5
    DD_BATCH_SIZE = 4096 # Set to large value to match PINN full-batch (N=64 => 4096 points)
    DD_SCALE_FACTOR = 100.0  # Scale solution for better training
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mesh type
    MESH_TYPE = "random"  # "grid" or "random"
    
    # Visualization
    PLOT_DPI = 150


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron base class"""
    
    def __init__(self, input_dim, output_dim, n_hidden_layers, width):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.width = width
        
        self.activation = nn.Tanh()
        self.input_layer = nn.Linear(self.input_dim, self.width)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.width, self.width) 
            for _ in range(self.n_hidden_layers - 1)
        ])
        self.output_layer = nn.Linear(self.width, self.output_dim)
        
        self.init_weights()
    
    def init_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


class DataDrivenModel(MLP):
    """Data-driven supervised learning model"""
    
    def __init__(self, n_hidden_layers, width):
        super(DataDrivenModel, self).__init__(2, 1, n_hidden_layers, width)
    
    def fit(self, training_set, num_epochs, optimizer, verbose=True):
        """Train the model on supervised data"""
        history = []
        
        for epoch in range(num_epochs):
            for input_coords, u_train in training_set:
                def closure():
                    optimizer.zero_grad()
                    u_pred = self(input_coords)
                    loss = torch.mean((u_pred.reshape(-1) - u_train.reshape(-1)) ** 2)
                    loss.backward()
                    history.append(loss.item())
                    return loss
                
                optimizer.step(closure=closure)
            
            if verbose and epoch % 50 == 0:
                if history:
                    print(f'Epoch {epoch:4d}, Loss: {history[-1]:.6e}')
        
        if verbose and history:
            print(f'Final Loss: {history[-1]:.6e}')
        
        return history


class PINN(MLP):
    """Physics-Informed Neural Network"""
    
    def __init__(self, n_hidden_layers, width, N, device, 
                 mesh="grid", lambda_u=1.0):
        super(PINN, self).__init__(2, 1, n_hidden_layers, width)
        
        self.device = device
        self.to(self.device)
        self.lambda_u = lambda_u
        self.N = N
        self.mesh = mesh
        
        self.domain_bounds = ([0.0, 1.0], [0.0, 1.0])
        self.space_dimension = 2
        self.num_boundary_points = self.N
        self.num_interior_points = (self.N - 2) ** 2
        
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.space_dimension)
        self.training_set_b, self.training_set_int = self.assemble_datasets()
    
    def add_boundary_points(self):
        """Generate boundary points"""
        if self.mesh == "grid":
            x_b = torch.linspace(self.domain_bounds[0][0], self.domain_bounds[0][1], 
                               self.N, device=self.device)
            y_b = torch.linspace(self.domain_bounds[1][0], self.domain_bounds[1][1], 
                               self.N, device=self.device)
            zero = torch.zeros_like(x_b)
            one = torch.ones_like(x_b)
            
            # Four sides of the boundary
            lower = torch.stack([x_b, zero], dim=1)
            upper = torch.stack([x_b, one], dim=1)
            left = torch.stack([zero, y_b], dim=1)
            right = torch.stack([one, y_b], dim=1)
            
            boundary_points = torch.cat([lower, upper, left, right], dim=0)
            
        elif self.mesh == "random":
            # Random boundary points using Sobol sequences
            samples = self.soboleng.draw(self.num_boundary_points).to(self.device)
            x_b = samples[:, 0]
            y_b = samples[:, 1]
            
            x_b = x_b * (self.domain_bounds[0][1] - self.domain_bounds[0][0]) + self.domain_bounds[0][0]
            y_b = y_b * (self.domain_bounds[1][1] - self.domain_bounds[1][0]) + self.domain_bounds[1][0]
            
            zero = torch.zeros_like(x_b)
            one = torch.ones_like(x_b)
            
            lower = torch.stack([x_b, zero], dim=1)
            upper = torch.stack([x_b, one], dim=1)
            left = torch.stack([zero, y_b], dim=1)
            right = torch.stack([one, y_b], dim=1)
            
            boundary_points = torch.cat([lower, upper, left, right], dim=0)
        
        return boundary_points, torch.zeros((boundary_points.shape[0], 1), device=self.device)
    
    def add_interior_points(self):
        """Generate interior points"""
        if self.mesh == "grid":
            x = torch.linspace(self.domain_bounds[0][0], self.domain_bounds[0][1], 
                             self.N, device=self.device)[1:-1]
            y = torch.linspace(self.domain_bounds[1][0], self.domain_bounds[1][1], 
                             self.N, device=self.device)[1:-1]
            
            X, Y = torch.meshgrid(x, y, indexing='ij')
            interior_points = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
            
        elif self.mesh == "random":
            interior_points = self.soboleng.draw(self.num_interior_points).to(self.device)
            
            interior_points[:, 0] = interior_points[:, 0] * (self.domain_bounds[0][1] - self.domain_bounds[0][0]) + self.domain_bounds[0][0]
            interior_points[:, 1] = interior_points[:, 1] * (self.domain_bounds[1][1] - self.domain_bounds[1][0]) + self.domain_bounds[1][0]
        
        return interior_points, torch.zeros((interior_points.shape[0], 1), device=self.device)
    
    def assemble_datasets(self):
        """Create boundary and interior dataloaders"""
        input_b, output_b = self.add_boundary_points()
        input_int, output_int = self.add_interior_points()
        
        training_set_b = DataLoader(
            torch.utils.data.TensorDataset(input_b, output_b),
            batch_size=input_b.shape[0],
            shuffle=False
        )
        training_set_int = DataLoader(
            torch.utils.data.TensorDataset(input_int, output_int),
            batch_size=input_int.shape[0],
            shuffle=False
        )
        
        return training_set_b, training_set_int
    
    def apply_boundary_conditions(self, input_b):
        """Apply boundary conditions"""
        return self(input_b)
    
    def compute_pde_residual(self, input_int, forcing_term):
        """Compute PDE residual: -Δu - f"""
        input_int.requires_grad = True
        u = self(input_int)
        
        # First derivatives
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_x = grad_u[:, 0]
        grad_u_y = grad_u[:, 1]
        
        # Second derivatives
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[0][:, 0]
        grad_u_yy = torch.autograd.grad(grad_u_y.sum(), input_int, create_graph=True)[0][:, 1]
        
        laplacian_u = grad_u_xx + grad_u_yy
        residual = -laplacian_u - forcing_term(input_int)
        
        return residual
    
    def compute_loss(self, inp_train_b, u_train_b, inp_train_int, forcing_term):
        """Compute total PINN loss"""
        u_pred_b = self.apply_boundary_conditions(inp_train_b)
        r_b = u_pred_b.reshape(-1) - u_train_b.reshape(-1)
        r_int = self.compute_pde_residual(inp_train_int, forcing_term)
        
        loss_b = torch.mean(r_b ** 2)
        loss_int = torch.mean(r_int ** 2)
        total_loss = loss_b + self.lambda_u * loss_int
        
        return total_loss
    
    def fit(self, num_epochs, optimizer, forcing_term, verbose=True):
        """Train the PINN"""
        history = []
        
        for epoch in range(num_epochs):
            for (inp_train_b, u_train_b), (inp_train_int, _) in zip(
                self.training_set_b, self.training_set_int
            ):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_b, u_train_b, 
                                           inp_train_int, forcing_term)
                    loss.backward()
                    history.append(loss.item())
                    return loss
                
                optimizer.step(closure=closure)
            
            if verbose and epoch % 50 == 0:
                if history:
                    print(f'Epoch {epoch:4d}, Loss: {history[-1]:.6e}')
        
        if verbose and history:
            print(f'Final Loss: {history[-1]:.6e}')
        
        return history


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def create_optimizers_pinn(model, config):
    """Create Adam and LBFGS optimizers for PINN"""
    optimizer_adam = optim.Adam(model.parameters(), lr=config.PINN_LR_ADAM)
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=config.PINN_LR_LBFGS,
        max_iter=50,
        max_eval=5000,
        history_size=150,
        line_search_fn="strong_wolfe",
        tolerance_change=1.0 * np.finfo(float).eps
    )
    return optimizer_adam, optimizer_lbfgs


def create_optimizers_dd(model, config):
    """Create Adam and LBFGS optimizers for data-driven model"""
    optimizer_adam = optim.Adam(model.parameters(), lr=config.DD_LR_ADAM)
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=config.DD_LR_LBFGS,
        max_iter=50,
        max_eval=5000,
        history_size=150,
        line_search_fn="strong_wolfe",
        tolerance_change=1.0 * np.finfo(float).eps
    )
    return optimizer_adam, optimizer_lbfgs


def create_forcing_term_function(data_generator, device):
    """Create forcing term callable from data generator"""
    def forcing_term(x):
        x_np = x[:, 0].detach().cpu().numpy()
        y_np = x[:, 1].detach().cpu().numpy()
        f_np = data_generator.forcing_term(x_np, y_np)
        return torch.tensor(f_np, dtype=torch.float32, device=device).reshape(-1)
    
    return forcing_term


def train_pinn(config, data_generator, verbose=True):
    """Train PINN model with two-stage optimization"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training PINN (K={config.K}, N={config.N})")
        print(f"{'='*60}")
    
    # Initialize model
    pinn = PINN(
        config.N_HIDDEN_LAYERS, 
        config.WIDTH, 
        config.N, 
        config.DEVICE,
        mesh=config.MESH_TYPE,
        lambda_u=config.PINN_LAMBDA_U
    )
    
    # Create forcing term
    forcing_term = create_forcing_term_function(data_generator, config.DEVICE)
    
    # Create optimizers
    optimizer_adam, optimizer_lbfgs = create_optimizers_pinn(pinn, config)
    
    # Stage 1: Adam pre-training
    if verbose:
        print("\nStage 1: Adam pre-training")
    hist_adam = pinn.fit(
        num_epochs=config.PINN_EPOCHS_ADAM,
        optimizer=optimizer_adam,
        forcing_term=forcing_term,
        verbose=verbose
    )
    
    # Stage 2: LBFGS fine-tuning
    if verbose:
        print("\nStage 2: LBFGS fine-tuning")
    hist_lbfgs = pinn.fit(
        num_epochs=config.PINN_EPOCHS_LBFGS,
        optimizer=optimizer_lbfgs,
        forcing_term=forcing_term,
        verbose=verbose
    )
    
    history = hist_adam + hist_lbfgs
    
    return pinn, history


def train_data_driven(config, data_generator, verbose=True):
    """Train data-driven model with supervised learning"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Data-Driven Model (K={config.K}, N={config.N})")
        print(f"{'='*60}")
    
    # Initialize model
    model = DataDrivenModel(config.N_HIDDEN_LAYERS, config.WIDTH).to(config.DEVICE)
    
    # Generate exact solution on grid
    x = np.linspace(0, 1, config.N)
    y = np.linspace(0, 1, config.N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    sol = data_generator.exact_solution(X, Y)  # Use the existing coefficients
    
    # Prepare training dataset
    coords = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1),
        dtype=torch.float32,
        device=config.DEVICE
    )
    targets = torch.tensor(
        sol.flatten() * config.DD_SCALE_FACTOR,
        dtype=torch.float32,
        device=config.DEVICE
    )
    
    # Use full batch size to match PINN training schedule
    batch_size = coords.shape[0]
    
    training_set = DataLoader(
        torch.utils.data.TensorDataset(coords, targets),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create optimizers
    optimizer_adam, optimizer_lbfgs = create_optimizers_dd(model, config)
    
    # Stage 1: Adam training
    if verbose:
        print("\nStage 1: Adam training")
    hist_adam = model.fit(
        training_set,
        num_epochs=config.DD_EPOCHS_ADAM,
        optimizer=optimizer_adam,
        verbose=verbose
    )
    
    # Stage 2: LBFGS fine-tuning
    if verbose:
        print("\nStage 2: LBFGS fine-tuning")
    hist_lbfgs = model.fit(
        training_set,
        num_epochs=config.DD_EPOCHS_LBFGS,
        optimizer=optimizer_lbfgs,
        verbose=verbose
    )
    
    history = hist_adam + hist_lbfgs
    
    return model, history


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_model(model, data_generator, config, is_data_driven=False):
    """Evaluate model and compute L2 relative error"""
    # Create evaluation grid
    x = np.linspace(0, 1, config.N)
    y = np.linspace(0, 1, config.N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Prepare input
    coords = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1),
        dtype=torch.float32,
        device=config.DEVICE
    )
    
    # Get predictions
    with torch.no_grad():
        U_pred = model(coords).detach().cpu().numpy().reshape(config.N, config.N)
    
    # Rescale if data-driven model
    if is_data_driven and hasattr(config, 'DD_SCALE_FACTOR'):
        U_pred = U_pred / config.DD_SCALE_FACTOR
    
    # Get exact solution
    U_exact = data_generator.exact_solution(X, Y)
    
    # Compute L2 relative error
    error = (
        torch.mean((torch.tensor(U_pred) - torch.tensor(U_exact)) ** 2) /
        torch.mean(torch.tensor(U_exact) ** 2)
    ) ** 0.5 * 100
    
    return U_pred, U_exact, error.item()


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_training_history(history, title, config):
    """Plot training loss history"""
    plt.figure(dpi=config.PLOT_DPI)
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(history) + 1), history, label="Training Loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_solution_comparison(U_pred, U_exact, error, title, config):
    """Plot predicted vs exact solution"""
    x = np.linspace(0, 1, config.N)
    y = np.linspace(0, 1, config.N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Predicted solution
    im0 = axes[0].pcolormesh(X, Y, U_pred, shading='auto')
    axes[0].set_title("Predicted Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])
    
    # Exact solution
    im1 = axes[1].pcolormesh(X, Y, U_exact, shading='auto')
    axes[1].set_title("Exact Solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1])
    
    # Absolute error
    error_abs = np.abs(U_pred - U_exact)
    im2 = axes[2].pcolormesh(X, Y, error_abs, shading='auto', cmap='Reds')
    axes[2].set_title(f"Absolute Error\nL2 Rel. Error: {error:.2f}%")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(K, N=64, model_type="both"):
    """
    Run experiment for given complexity K and resolution N
    
    Parameters:
    -----------
    K : int
        Frequency complexity (1, 4, 8, 16)
    N : int
        Grid resolution
    model_type : str
        "pinn", "data_driven", or "both"
    """
    # Update configuration
    config = Config()
    config.K = K
    config.N = N
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: K={K}, N={N}")
    print(f"{'='*70}")
    print(f"Device: {config.DEVICE}")
    
    # Generate data
    data_generator = Poisson_data_generator(config.N, config.K)
    force, sol = data_generator.generate()
    
    results = {}
    
    # Train PINN
    if model_type in ["pinn", "both"]:
        pinn, hist_pinn = train_pinn(config, data_generator, verbose=True)
        U_pred_pinn, U_exact, error_pinn = evaluate_model(
            pinn, data_generator, config, is_data_driven=False
        )
        
        results['pinn'] = {
            'model': pinn,
            'history': hist_pinn,
            'prediction': U_pred_pinn,
            'error': error_pinn
        }
        
        print(f"\nPINN L2 Relative Error: {error_pinn:.2f}%")
        
        # Visualizations
        plot_training_history(
            hist_pinn,
            f"PINN Training History (K={K}, N={N})",
            config
        )
        
        plot_solution_comparison(
            U_pred_pinn, U_exact, error_pinn,
            f"PINN Results (K={K}, N={N})",
            config
        )
    
    # Train Data-Driven
    if model_type in ["data_driven", "both"]:
        dd_model, hist_dd = train_data_driven(config, data_generator, verbose=True)
        U_pred_dd, U_exact, error_dd = evaluate_model(
            dd_model, data_generator, config, is_data_driven=True
        )
        
        results['data_driven'] = {
            'model': dd_model,
            'history': hist_dd,
            'prediction': U_pred_dd,
            'error': error_dd
        }
        
        print(f"\nData-Driven L2 Relative Error: {error_dd:.2f}%")
        
        # Visualizations
        plot_training_history(
            hist_dd,
            f"Data-Driven Training History (K={K}, N={N})",
            config
        )
        
        plot_solution_comparison(
            U_pred_dd, U_exact, error_dd,
            f"Data-Driven Results (K={K}, N={N})",
            config
        )
    
    plt.show()
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Single experiment
    # results = run_experiment(K=4, N=64, model_type="both")
    
    # Multiple complexity levels
    complexity_levels = [1, 4, 8, 16]
    all_results = {}
    
    for K in complexity_levels:
        all_results[K] = run_experiment(K=K, N=64, model_type="both")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*70}")
    print(f"{'K':>5} | {'PINN Error (%)':>20} | {'Data-Driven Error (%)':>20}")
    print(f"{'-'*70}")
    
    for K in complexity_levels:
        pinn_err = all_results[K]['pinn']['error']
        dd_err = all_results[K]['data_driven']['error']
        print(f"{K:5d} | {pinn_err:20.2f} | {dd_err:20.2f}")


if __name__ == "__main__":
    main()