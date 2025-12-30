
import sys
import os
import glob
import matplotlib.pyplot as plt

# Add loss-landscape-master to path
sys.path.append(os.path.join(os.getcwd(), 'loss-landscape-master'))

import plot_2D

import h5py
import numpy as np

def replot_all():
    results_dir = "results/loss_landscapes_adapted"
    
    # Find all surface files
    surface_files = glob.glob(os.path.join(results_dir, "*_surface.h5"))
    
    print(f"Found {len(surface_files)} surface files to replot.")
    
    # First pass: Find global min and max
    global_min = float('inf')
    global_max = float('-inf')
    
    print("Calculating global bounds...")
    for surf_file in surface_files:
        try:
            with h5py.File(surf_file, 'r') as f:
                if 'train_loss' in f.keys():
                    data = np.array(f['train_loss'][:])
                    global_min = min(global_min, np.min(data))
                    global_max = max(global_max, np.max(data))
        except Exception as e:
            print(f"Error reading {surf_file}: {e}")

    print(f"Global Min: {global_min}")
    print(f"Global Max: {global_max}")
    
    # Second pass: Plot with global bounds
    for surf_file in surface_files:
        print(f"\nReplotting {surf_file} with global bounds...")
        plot_2D.plot_2d_contour(
            surf_file, 
            'train_loss', 
            vmin=global_min, 
            vmax=global_max, 
            vlevel=None, 
            show=False
        )
        print("Done.")

if __name__ == "__main__":
    replot_all()
