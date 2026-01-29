# Hessian Eigenvector Directions for Loss Landscape Visualization

## Overview

This implementation adds **Hessian eigenvector-based directions** to the loss landscape visualization, following the methodology in **Krishnapriyan et al. 2021** (Section 4.1):

> "We plot the loss landscape by perturbing the trained model across the first two dominant Hessian eigenvectors"

This replaces the random filter-normalized directions from Li et al. 2018 with **principal curvature directions** that reveal the true geometry of the loss surface.

---

## Key Files

### 1. `hessian_utils.py` (NEW)
Core implementation of Hessian eigenvector computation:

- **`compute_hessian_eigenvectors()`**: Main function to compute dominant eigenvectors
  - Uses **Lanczos algorithm** via `scipy.sparse.linalg.eigsh`
  - Efficient: computes Hessian-vector products without forming full Hessian matrix
  - Returns: max/min eigenvectors (as parameter lists) and eigenvalues

- **`hessian_vector_product()`**: Computes H·v via double backward pass
  - H·v = ∇(∇L·v) using PyTorch autograd
  - O(n) complexity instead of O(n²) for full Hessian

- **`filter_normalize_direction()`**: Applies Li et al. 2018 filter normalization
  - Ensures fair comparison between Hessian and random directions
  - Rescales each filter to match model parameter norms

### 2. `task3_loss_landscape.py` (MODIFIED)
Updated to support Hessian directions:

- **LandscapeConfig**: New options
  - `USE_HESSIAN_DIRECTIONS = True`: Toggle between Hessian/random directions
  - `HESSIAN_MAXITER = 100`: Lanczos iteration limit
  - `HESSIAN_TOL = 1e-3`: Convergence tolerance

- **LandscapeAdapter.generate_directions()**: New method
  - Computes or loads Hessian eigenvectors
  - Applies filter normalization
  - Saves directions to `.h5` files with `_hessian` suffix
  - Stores eigenvalue info for plotting

- **Updated plotting**:
  - Axis labels: "max eigenvector" vs "min eigenvector"
  - Titles include λ_max and λ_min values
  - Filenames: `comparative_loss_landscapes_hessian_vmax*.pdf`

### 3. `test_hessian.py` (NEW)
Test suite to verify implementation:

- **Test 1**: Simple quadratic loss (known Hessian)
- **Test 2**: Filter normalization correctness
- **Test 3**: PINN-like loss with PDE residuals

---

## Mathematical Background

### Hessian Matrix
The Hessian H of loss L(θ) at trained parameters θ*:

$$H = \nabla^2 L(\theta^*) = \left[\frac{\partial^2 L}{\partial \theta_i \partial \theta_j}\right]$$

### Dominant Eigenvectors
We compute eigenvectors for:
- **λ_max**: Largest positive eigenvalue (steepest ascent direction)
- **λ_min**: Most negative eigenvalue (steepest descent direction)

These represent the **principal curvatures** of the loss surface.

### Why Hessian Directions?
- **Random directions** (Li et al. 2018): May miss interesting geometry
- **Hessian eigenvectors**: Capture true curvature, reveal:
  - Sharp vs flat dimensions
  - Saddle points and non-convexity
  - Failure modes in PINNs (high K)

---

## Usage

### Quick Start

1. **Test the implementation:**
   ```bash
   python test_hessian.py
   ```

2. **Run loss landscape with Hessian directions:**
   ```bash
   python task3_loss_landscape.py
   ```

3. **Switch to random directions (for comparison):**
   Edit `task3_loss_landscape.py`:
   ```python
   class LandscapeConfig:
       USE_HESSIAN_DIRECTIONS = False  # Change to False
   ```

### Configuration Options

In `task3_loss_landscape.py`:

```python
class LandscapeConfig:
    # Direction method
    USE_HESSIAN_DIRECTIONS = True    # True: Hessian, False: Random
    
    # Hessian computation
    HESSIAN_MAXITER = 100            # Lanczos iterations (increase for more accuracy)
    HESSIAN_TOL = 1e-3               # Convergence tolerance (decrease for more precision)
    
    # Models to process
    K_LEVELS = [1, 4, 16]            # Frequency parameters
    MODELS = ['pinn', 'dd']          # Model types
    
    # Force recomputation
    FORCE_RECOMPUTE = True           # Set False to reuse cached directions
```

---

## Output

### Directory Structure
```
results/loss_landscapes_adapted/
├── pinn_K1_directions_hessian.h5     # Hessian eigenvectors (saved)
├── pinn_K1_surface.h5                # Loss landscape data
├── ...
└── comparative_loss_landscapes_hessian_vmax10.pdf  # Final plots
```

### H5 File Contents
Direction files include:
- `xdirection`: Maximum eigenvalue eigenvector (filter-normalized)
- `ydirection`: Minimum eigenvalue eigenvector (filter-normalized)
- `eigenvalue_max`: λ_max value
- `eigenvalue_min`: λ_min value
- `dot_product`: Orthogonality check (should be ≈0)

### Plot Features
- **Axis labels**: Indicate direction type (Hessian eigenvectors)
- **Titles**: Show eigenvalues (λ_max, λ_min) for each subplot
- **Filenames**: Include `_hessian` or `_random` suffix

---

## Implementation Details

### Computational Efficiency

**Challenge**: Full Hessian matrix has O(n²) elements (n = # parameters)
- Example: 50K parameters → 2.5B Hessian elements (10 GB memory!)

**Solution**: Implicit Hessian-vector products
1. Compute loss gradient: g = ∇L(θ)
2. For direction v, compute: H·v = ∇(g·v) using autograd
3. Use Lanczos algorithm with matrix-free operator

**Complexity**:
- Full Hessian: O(n² × data_size)
- Our method: O(k × n × data_size) where k = # iterations ≪ n

### Numerical Stability

- **Normalization**: Apply same filter normalization as random directions
- **Tolerance**: Use `tol=1e-3` for balance of speed/accuracy
- **Bias/BN handling**: Ignore 1D parameters (consistent with random directions)

### Edge Cases Handled

1. **eigsh fails**: Automatically falls back to random directions
2. **Non-convergence**: Increase `HESSIAN_MAXITER` or reduce `HESSIAN_TOL`
3. **Memory issues**: Reduce grid resolution (`N` in Config)

---

## Expected Results

### Well-Trained Models (Low K)
- **Hessian-based plots**: Show convex bowl-like structure
- λ_max and |λ_min| should be similar order of magnitude
- Smoother loss surface along eigenvector directions

### Failed Models (High K)
- **Hessian-based plots**: Reveal chaotic non-convexity
- Large eigenvalue spread: |λ_max| ≫ |λ_min| or vice versa
- Multiple local minima and saddle points visible

### Comparison to Random Directions
- Random directions often show more uniform structure
- Hessian directions emphasize **anisotropic** curvature
- More pronounced differences at high K

---

## Troubleshooting

### Issue: "eigsh failed" error

**Cause**: Lanczos didn't converge

**Solutions**:
1. Increase iterations: `HESSIAN_MAXITER = 200`
2. Relax tolerance: `HESSIAN_TOL = 1e-2`
3. Check if loss is differentiable (requires `create_graph=True`)

### Issue: Very long computation time

**Cause**: Hessian computation is expensive

**Solutions**:
1. Reduce grid resolution: `N = 32` (instead of 64)
2. Reduce landscape resolution: `XNUM = 50` (instead of 100)
3. First run test: Set `K_LEVELS = [1]` and `MODELS = ['pinn']`

### Issue: "Orthogonality check failed"

**Cause**: Eigenvectors not orthogonal (numerical error)

**Impact**: Usually benign if |dot_product| < 1e-3
**Fix**: Increase `HESSIAN_TOL` to 1e-4 for more precision

---

## References

1. **Krishnapriyan et al. 2021**: "Characterizing possible failure modes in physics-informed neural networks"
   - Section 4.1: Loss landscape analysis with Hessian eigenvectors

2. **Li et al. 2018**: "Visualizing the Loss Landscape of Neural Nets"
   - Filter normalization for fair direction comparison

3. **Yao et al. 2020**: "PyHessian: Neural Networks Through the Lens of the Hessian"
   - Efficient Hessian computation techniques

---

## Testing Checklist

Before running full experiments:

- [ ] Run `python test_hessian.py` → All tests pass
- [ ] Test on K=1 only (fast) → Generates plots
- [ ] Check `.h5` files contain eigenvalue data
- [ ] Verify plot labels show "max/min eigenvector"
- [ ] Compare Hessian vs random directions (toggle config)
- [ ] Confirm orthogonality: |v_max · v_min| < 0.01

---

## Performance Notes

**Typical timing** (N=64, 100×100 landscape, CPU):
- Random directions: ~5 min per model
- Hessian directions: ~15 min per model (10 min for eigenvectors + 5 min for surface)

**GPU acceleration**:
- Hessian computation: Yes (autograd on GPU)
- Lanczos algorithm: CPU only (scipy)
- Overall speedup: ~2x with GPU

---

## Future Enhancements

1. **Top-k eigenvectors**: Compute more than 2 directions for higher-dimensional visualization
2. **Subsampling**: Use subset of data for faster Hessian computation
3. **Block Hessian**: Compute Hessian per-layer for very large networks
4. **Curvature tracking**: Plot eigenvalues during training (not just at convergence)

---

## Contact

For questions or issues with this implementation, refer to:
- Original paper: Krishnapriyan et al. 2021 (arXiv:2109.01050)
- Loss landscape library: https://github.com/tomgoldstein/loss-landscape
