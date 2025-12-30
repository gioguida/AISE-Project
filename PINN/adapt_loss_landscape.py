
import sys
import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from argparse import Namespace
import copy

# Add loss-landscape-master to path
sys.path.append(os.path.join(os.getcwd(), 'loss-landscape-master'))

# Import loss-landscape-master modules
# We need to mock mpi4pytorch if it's not available or just avoid using it
try:
    import mpi4pytorch as mpi
except ImportError:
    mpi = None

import net_plotter
import plot_2D
import h5_util

# Import project modules
from task2_implementation import (
    Config, 
    Poisson_data_generator,
    PINN,
    DataDrivenModel
)

class LandscapeAdapter:
    def __init__(self):
        self.config = Config()
        self.config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = "results/loss_landscapes_adapted"
        os.makedirs(self.results_dir, exist_ok=True)

    def get_pinn_loss(self, net, data_generator):
        # Generate a fixed set of points for evaluation
        # We use the grid points for consistency
        x = np.linspace(0, 1, self.config.N)
        y = np.linspace(0, 1, self.config.N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Interior points (using all grid points for landscape smoothness)
        inp = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                          dtype=torch.float32, 
                          device=self.config.DEVICE).requires_grad_(True)
        
        # Forcing term
        f_val = data_generator.forcing_term(inp[:, 0].detach().cpu().numpy(), 
                                          inp[:, 1].detach().cpu().numpy())
        f_val = torch.tensor(f_val, dtype=torch.float32, device=self.config.DEVICE)
        
        # Compute PDE residual
        u = net(inp)
        
        grads = torch.autograd.grad(u, inp, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grads[:, 0]
        u_y = grads[:, 1]
        
        grads_x = torch.autograd.grad(u_x, inp, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xx = grads_x[:, 0]
        
        grads_y = torch.autograd.grad(u_y, inp, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_yy = grads_y[:, 1]
        
        residual = -(u_xx + u_yy) - f_val
        loss_pde = torch.mean(residual**2)
        
        # We focus on PDE residual for the landscape as it's the most interesting part
        return loss_pde.item()

    def get_dd_loss(self, net, data_generator):
        _, solution = data_generator.generate()
        
        x = np.linspace(0, 1, self.config.N)
        y = np.linspace(0, 1, self.config.N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inp = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), 
                          dtype=torch.float32, 
                          device=self.config.DEVICE)
        
        target = torch.tensor(solution.flatten() * self.config.DD_SCALE_FACTOR, 
                             dtype=torch.float32, 
                             device=self.config.DEVICE)
        
        pred = net(inp).reshape(-1)
        loss = torch.mean((pred - target)**2)
        return loss.item()

    def crunch_surface(self, surf_file, net, w, d, loss_fn, args):
        print(f"Computing surface for {surf_file}...")
        
        # Create/Open h5 file
        f = h5py.File(surf_file, 'w')
        f['dir_file'] = args.dir_file
        
        xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
        ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
        
        f['xcoordinates'] = xcoordinates
        f['ycoordinates'] = ycoordinates
        
        losses = np.zeros((len(xcoordinates), len(ycoordinates)))
        
        # Loop over coordinates
        total = len(xcoordinates) * len(ycoordinates)
        count = 0
        
        for i, x in enumerate(xcoordinates):
            for j, y in enumerate(ycoordinates):
                # Set weights: w + x*d[0] + y*d[1]
                net_plotter.set_weights(net, w, d, step=[x, y])
                
                # Evaluate loss
                losses[i, j] = loss_fn()
                
                count += 1
                if count % 100 == 0:
                    print(f"  Progress: {count}/{total}", end='\r')
        
        print(f"  Progress: {total}/{total}")
        
        f['train_loss'] = losses
        f.close()
        
        # Restore weights
        net_plotter.set_weights(net, w)

    def run(self):
        K_LEVELS = [1, 4, 16]
        
        for K in K_LEVELS:
            print(f"\nProcessing K={K}...")
            self.config.K = K
            data_generator = Poisson_data_generator(self.config.N, K)
            
            # ---------------------------------------------------------
            # PINN
            # ---------------------------------------------------------
            pinn_path = f"results/models/pinn_K{K}.pt"
            if os.path.exists(pinn_path):
                print(f"Loading PINN from {pinn_path}")
                pinn_model = PINN(
                    self.config.N_HIDDEN_LAYERS, 
                    self.config.WIDTH, 
                    self.config.N, 
                    self.config.DEVICE,
                    mesh=self.config.MESH_TYPE,
                    lambda_u=self.config.PINN_LAMBDA_U
                )
                pinn_model.load_state_dict(torch.load(pinn_path, map_location=self.config.DEVICE))
                
                # Setup args for net_plotter
                args = Namespace(
                    dir_file=os.path.join(self.results_dir, f"pinn_K{K}_directions.h5"),
                    dir_type='weights',
                    xignore='biasbn',
                    xnorm='filter',
                    yignore='biasbn',
                    ynorm='filter',
                    same_dir=False,
                    model_file2=None,
                    model_file3=None,
                    y=True,
                    idx=0,
                    xmin=-1.0, xmax=1.0, xnum=50,
                    ymin=-1.0, ymax=1.0, ynum=50
                )
                
                # Generate directions manually
                print("Generating directions...")
                xdirection = net_plotter.create_random_direction(pinn_model, args.dir_type, args.xignore, args.xnorm)
                ydirection = net_plotter.create_random_direction(pinn_model, args.dir_type, args.yignore, args.ynorm)
                
                # Write to h5
                f = h5py.File(args.dir_file, 'w')
                h5_util.write_list(f, 'xdirection', xdirection)
                h5_util.write_list(f, 'ydirection', ydirection)
                f.close()
                
                # Load directions
                directions = [xdirection, ydirection]
                
                # Compute surface
                surf_file = os.path.join(self.results_dir, f"pinn_K{K}_surface.h5")
                w = net_plotter.get_weights(pinn_model)
                
                self.crunch_surface(
                    surf_file, 
                    pinn_model, 
                    w, 
                    directions, 
                    lambda: self.get_pinn_loss(pinn_model, data_generator),
                    args
                )
                
                # Plot
                print("Plotting PINN surface...")
                plot_2D.plot_2d_contour(surf_file, 'train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False)
                
            
            # ---------------------------------------------------------
            # Data Driven
            # ---------------------------------------------------------
            dd_path = f"results/models/data_driven_K{K}.pt"
            if os.path.exists(dd_path):
                print(f"Loading Data-Driven from {dd_path}")
                dd_model = DataDrivenModel(self.config.N_HIDDEN_LAYERS, self.config.WIDTH).to(self.config.DEVICE)
                dd_model.load_state_dict(torch.load(dd_path, map_location=self.config.DEVICE))
                
                # Setup args
                args = Namespace(
                    dir_file=os.path.join(self.results_dir, f"dd_K{K}_directions.h5"),
                    dir_type='weights',
                    xignore='biasbn',
                    xnorm='filter',
                    yignore='biasbn',
                    ynorm='filter',
                    same_dir=False,
                    model_file2=None,
                    model_file3=None,
                    y=True,
                    idx=0,
                    xmin=-1.0, xmax=1.0, xnum=50,
                    ymin=-1.0, ymax=1.0, ynum=50
                )
                
                # Generate directions manually
                print("Generating directions...")
                xdirection = net_plotter.create_random_direction(dd_model, args.dir_type, args.xignore, args.xnorm)
                ydirection = net_plotter.create_random_direction(dd_model, args.dir_type, args.yignore, args.ynorm)
                
                # Write to h5
                f = h5py.File(args.dir_file, 'w')
                h5_util.write_list(f, 'xdirection', xdirection)
                h5_util.write_list(f, 'ydirection', ydirection)
                f.close()
                
                # Load directions
                directions = [xdirection, ydirection]
                
                # Compute surface
                surf_file = os.path.join(self.results_dir, f"dd_K{K}_surface.h5")
                w = net_plotter.get_weights(dd_model)
                
                self.crunch_surface(
                    surf_file, 
                    dd_model, 
                    w, 
                    directions, 
                    lambda: self.get_dd_loss(dd_model, data_generator),
                    args
                )
                
                # Plot
                print("Plotting Data-Driven surface...")
                plot_2D.plot_2d_contour(surf_file, 'train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False)

if __name__ == "__main__":
    adapter = LandscapeAdapter()
    adapter.run()
