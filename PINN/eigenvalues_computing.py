"""
Compute Hessian eigenvalues for PINN and Data-Driven models at all complexity levels.

This script computes the dominant eigenvalues (maximum and minimum) of the Hessian
matrix for the loss function at the trained model parameters. These eigenvalues
characterize the curvature of the loss landscape.

Results are saved to a JSON file for analysis.
"""

import sys
import os
import torch
import numpy as np

# Add loss-landscape-master to path
sys.path.append(os.path.join(os.getcwd(), 'loss-landscape-master'))

# Import project modules
from task2_implementation import (
    Config, 
    Poisson_data_generator,
    PINN,
    DataDrivenModel
)
import hessian_utils


class EigenvalueComputer:
    def __init__(self):
        self.config = Config()
        self.config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = "results/models"
        self.results = {}
        
        # Hessian computation parameters
        self.maxiter = 100
        self.tol = 1e-3
        
        print(f"Using device: {self.config.DEVICE}")
        print(f"Models directory: {self.models_dir}")
        print(f"Hessian computation: maxiter={self.maxiter}, tol={self.tol}")
    
    def get_pinn_loss_function(self, model, data_generator):
        """Create a closure that computes PINN loss (PDE residual)."""
        def loss_fn():
            # Generate evaluation points
            x = np.linspace(0, 1, self.config.N)
            y = np.linspace(0, 1, self.config.N)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # Interior points
            inp = torch.tensor(
                np.stack([X.flatten(), Y.flatten()], axis=1), 
                dtype=torch.float32, 
                device=self.config.DEVICE
            ).requires_grad_(True)
            
            # Forcing term
            f_val = data_generator.forcing_term(
                inp[:, 0].detach().cpu().numpy(), 
                inp[:, 1].detach().cpu().numpy()
            )
            f_val = torch.tensor(f_val, dtype=torch.float32, device=self.config.DEVICE)
            
            # Compute PDE residual
            u = model(inp)
            
            grads = torch.autograd.grad(
                u, inp, 
                grad_outputs=torch.ones_like(u), 
                create_graph=True
            )[0]
            u_x = grads[:, 0]
            u_y = grads[:, 1]
            
            grads_x = torch.autograd.grad(
                u_x, inp, 
                grad_outputs=torch.ones_like(u_x), 
                create_graph=True
            )[0]
            u_xx = grads_x[:, 0]
            
            grads_y = torch.autograd.grad(
                u_y, inp, 
                grad_outputs=torch.ones_like(u_y), 
                create_graph=True
            )[0]
            u_yy = grads_y[:, 1]
            
            residual = -(u_xx + u_yy) - f_val
            loss_pde = torch.mean(residual**2)
            
            return loss_pde
        
        return loss_fn
    
    def get_dd_loss_function(self, model, data_generator):
        """Create a closure that computes Data-Driven loss (MSE)."""
        def loss_fn():
            _, solution = data_generator.generate()
            
            x = np.linspace(0, 1, self.config.N)
            y = np.linspace(0, 1, self.config.N)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            inp = torch.tensor(
                np.stack([X.flatten(), Y.flatten()], axis=1), 
                dtype=torch.float32, 
                device=self.config.DEVICE
            )
            
            target = torch.tensor(
                solution.flatten() * self.config.DD_SCALE_FACTOR, 
                dtype=torch.float32, 
                device=self.config.DEVICE
            )
            
            pred = model(inp).reshape(-1)
            loss = torch.mean((pred - target)**2)
            
            return loss
        
        return loss_fn
    
    def compute_eigenvalues_for_model(self, model_type, K):
        """
        Compute eigenvalues for a specific model.
        
        Args:
            model_type: 'pinn' or 'dd'
            K: complexity level (1, 4, or 16)
        
        Returns:
            dict with eigenvalue information or None if failed
        """
        print(f"\n{'='*70}")
        print(f"Computing eigenvalues for {model_type.upper()} with K={K}")
        print(f"{'='*70}")
        
        # Setup
        self.config.K = K
        data_generator = Poisson_data_generator(self.config.N, K)
        
        # Load model
        if model_type == 'pinn':
            model_path = os.path.join(self.models_dir, f"pinn_K{K}.pt")
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found: {model_path}")
                return None
            
            model = PINN(
                self.config.N_HIDDEN_LAYERS, 
                self.config.WIDTH, 
                self.config.N, 
                self.config.DEVICE,
                mesh=self.config.MESH_TYPE,
                lambda_u=self.config.PINN_LAMBDA_U
            )
            model.load_state_dict(torch.load(model_path, map_location=self.config.DEVICE))
            loss_fn = self.get_pinn_loss_function(model, data_generator)
            
        elif model_type == 'dd':
            model_path = os.path.join(self.models_dir, f"data_driven_K{K}.pt")
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found: {model_path}")
                return None
            
            model = DataDrivenModel(
                self.config.N_HIDDEN_LAYERS, 
                self.config.WIDTH
            ).to(self.config.DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=self.config.DEVICE))
            loss_fn = self.get_dd_loss_function(model, data_generator)
        else:
            print(f"ERROR: Unknown model type: {model_type}")
            return None
        
        print(f"Loaded model from: {model_path}")
        
        # Evaluate loss at trained parameters
        model.eval()
        loss_value = loss_fn().item()
        print(f"Loss at trained parameters: {loss_value:.6e}")
        
        # Compute Hessian eigenvalues
        print(f"\nComputing Hessian eigenvalues...")
        result = hessian_utils.compute_hessian_eigenvectors(
            model,
            loss_fn,
            num_eigenvalues=2,
            maxiter=self.maxiter,
            tol=self.tol
        )
        
        if result is None:
            print("ERROR: Hessian computation failed")
            return None
        
        # Extract results
        eigenvalue_max = float(result['eigenvalue_max'])
        eigenvalue_min = float(result['eigenvalue_min'])
        condition_number = float(abs(eigenvalue_max) / abs(eigenvalue_min))
        
        print(f"lambda_max = {eigenvalue_max:.6e}")
        print(f"lambda_min = {eigenvalue_min:.6e}")
        print(f"condition_number = {condition_number:.6e}")
        
        eigenvalue_info = {
            'eigenvalue_max': eigenvalue_max,
            'eigenvalue_min': eigenvalue_min,
            'condition_number': condition_number,
            'loss_value': float(loss_value)
        }
        
        return eigenvalue_info
    
    def run(self):
        """Compute eigenvalues for all models at all complexity levels."""
        K_levels = [1, 4, 16]
        model_types = ['pinn', 'dd']
        
        # Compute for all combinations
        for model_type in model_types:
            self.results[model_type] = {}
            for K in K_levels:
                result = self.compute_eigenvalues_for_model(model_type, K)
                if result is not None:
                    self.results[model_type][f'K{K}'] = result
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save eigenvalue results to text file."""
        output_file = "results/eigenvalues.txt"
        
        with open(output_file, 'w') as f:
            for model_type in ['pinn', 'dd']:
                if model_type not in self.results:
                    continue
                
                for K_key in ['K1', 'K4', 'K16']:
                    if K_key not in self.results[model_type]:
                        continue
                    
                    info = self.results[model_type][K_key]
                    K_val = K_key[1:]
                    
                    f.write(f"{model_type}_K{K_val}:\n")
                    f.write(f"  lambda_max = {info['eigenvalue_max']:.6e}\n")
                    f.write(f"  lambda_min = {info['eigenvalue_min']:.6e}\n")
                    f.write(f"  condition_number = {info['condition_number']:.6e}\n")
                    f.write(f"  loss = {info['loss_value']:.6e}\n")
                    f.write("\n")
        
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    computer = EigenvalueComputer()
    computer.run()
