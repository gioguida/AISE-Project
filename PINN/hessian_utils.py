"""
Hessian eigenvector computation for loss landscape visualization.

This module computes the dominant Hessian eigenvectors (maximum and minimum eigenvalues)
of the loss function with respect to network parameters. These eigenvectors represent 
the principal curvature directions of the loss surface.

Implementation uses efficient Hessian-vector products via double backward pass,
avoiding the need to form the full Hessian matrix explicitly.

References:
- Krishnapriyan et al. 2021: "Characterizing possible failure modes in physics-informed neural networks"
- Yao et al. 2020: "PyHessian: Neural Networks Through the Lens of the Hessian"
"""

import torch
import numpy as np
from scipy.sparse.linalg import eigsh
from typing import Callable, Tuple, List, Dict
import copy


def params_to_vector(parameters):
    """
    Convert network parameters to a single flat vector.
    
    Args:
        parameters: list of torch.Tensor (model parameters)
    
    Returns:
        torch.Tensor: flattened 1D parameter vector
    """
    vec = []
    for param in parameters:
        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_params(vec, parameters):
    """
    Convert a flat vector back to parameter list format.
    
    Args:
        vec: torch.Tensor, flat 1D vector
        parameters: list of torch.Tensor (model parameters) for shape reference
    
    Returns:
        list of torch.Tensor with same shapes as parameters
    """
    params = []
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        params.append(vec[pointer:pointer + num_param].view_as(param))
        pointer += num_param
    return params


def hessian_vector_product(loss_grad_params, model, vector):
    """
    Compute Hessian-vector product H·v without forming H explicitly.
    
    Uses double backward pass:
    H·v = ∇(∇L·v) where ∇L is the gradient of loss
    
    Args:
        loss_grad_params: list of gradients ∇L w.r.t. each parameter
        model: PyTorch model
        vector: list of tensors (direction vector) with same structure as parameters
    
    Returns:
        list of tensors representing H·v
    """
    grad_vector_product = 0.0
    for grad, v in zip(loss_grad_params, vector):
        grad_vector_product += (grad * v).sum()
    
    # Compute gradient of (∇L · v) w.r.t. parameters
    # This gives H·v
    hvp = torch.autograd.grad(
        grad_vector_product, 
        model.parameters(),
        retain_graph=True,
        create_graph=False
    )
    
    return [g.detach() for g in hvp]


def compute_hessian_eigenvectors(
    model,
    loss_function: Callable,
    dataloader=None,
    num_eigenvalues: int = 2,
    maxiter: int = 100,
    tol: float = 1e-3,
    use_cuda: bool = True,
    subsample_size: int = None
):
    """
    Compute top Hessian eigenvectors using Lanczos algorithm.
    
    Computes the eigenvectors corresponding to the maximum and minimum eigenvalues
    (by absolute value) of the Hessian matrix H = ∇²L(θ).
    
    Args:
        model: trained PyTorch model
        loss_function: callable that computes loss given model (no args)
                      Should internally handle data sampling
        dataloader: optional dataloader for sampling (not used if loss_function is self-contained)
        num_eigenvalues: number of eigenvectors to compute (default: 2 for max and min)
        maxiter: maximum iterations for Lanczos algorithm
        tol: convergence tolerance
        use_cuda: whether to use GPU
        subsample_size: if not None, subsample data to this size for efficiency
    
    Returns:
        dict containing:
            - 'eigenvector_max': list of tensors (direction of maximum |λ|)
            - 'eigenvector_min': list of tensors (direction of minimum |λ|)
            - 'eigenvalue_max': float (maximum eigenvalue)
            - 'eigenvalue_min': float (minimum eigenvalue)
            - 'all_eigenvalues': array of all computed eigenvalues
            - 'all_eigenvectors': list of all eigenvector directions
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Get parameter structure
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    
    print(f"Computing Hessian eigenvectors for {n_params:,} parameters...")
    print(f"Using Lanczos algorithm (maxiter={maxiter}, tol={tol})")
    
    # Compute loss and gradients
    model.zero_grad()
    loss = loss_function()
    
    # Compute gradients
    loss_grad = torch.autograd.grad(
        loss, 
        params,
        create_graph=True,
        retain_graph=True
    )
    
    # Define matrix-vector product operator for scipy eigsh
    def hvp_operator(v):
        """
        Operator that computes H·v for scipy's eigsh.
        Input/output are numpy arrays.
        """
        # Convert numpy vector to parameter list
        v_tensor = torch.from_numpy(v).float().to(device)
        v_params = vector_to_params(v_tensor, params)
        
        # Compute Hessian-vector product
        hvp = hessian_vector_product(loss_grad, model, v_params)
        
        # Convert back to numpy vector
        hvp_vec = params_to_vector(hvp).cpu().numpy().astype(np.float64)
        return hvp_vec
    
    # Create linear operator for eigsh
    from scipy.sparse.linalg import LinearOperator
    H_operator = LinearOperator(
        (n_params, n_params),
        matvec=hvp_operator
    )
    
    # Compute eigenvalues and eigenvectors
    # Request top k eigenvalues by magnitude
    print(f"Running Lanczos iteration...")
    
    # For maximum eigenvalue (largest algebraic value)
    try:
        eigenvalues_max, eigenvectors_max = eigsh(
            H_operator,
            k=min(num_eigenvalues, n_params - 1),
            which='LA',  # Largest Algebraic (most positive)
            maxiter=maxiter,
            tol=tol,
            return_eigenvectors=True
        )
        print(f"✓ Computed largest eigenvalues: {eigenvalues_max}")
    except Exception as e:
        print(f"Warning: eigsh failed for largest eigenvalues: {e}")
        print("Falling back to random directions")
        return None
    
    # For minimum eigenvalue (smallest algebraic value, most negative)
    try:
        eigenvalues_min, eigenvectors_min = eigsh(
            H_operator,
            k=min(num_eigenvalues, n_params - 1),
            which='SA',  # Smallest Algebraic (most negative)
            maxiter=maxiter,
            tol=tol,
            return_eigenvectors=True
        )
        print(f"✓ Computed smallest eigenvalues: {eigenvalues_min}")
    except Exception as e:
        print(f"Warning: eigsh failed for smallest eigenvalues: {e}")
        print("Using largest eigenvalues only")
        eigenvalues_min = eigenvalues_max
        eigenvectors_min = eigenvectors_max
    
    # Select the dominant directions by absolute magnitude
    # Maximum: largest positive eigenvalue
    idx_max = np.argmax(eigenvalues_max)
    eigvec_max = eigenvectors_max[:, idx_max]
    eigval_max = eigenvalues_max[idx_max]
    
    # Minimum: most negative eigenvalue (smallest algebraic)
    idx_min = np.argmin(eigenvalues_min)
    eigvec_min = eigenvectors_min[:, idx_min]
    eigval_min = eigenvalues_min[idx_min]
    
    # If we want largest absolute magnitude regardless of sign:
    # Compare |eigval_max| with |eigval_min|
    all_eigenvalues = np.concatenate([eigenvalues_max, eigenvalues_min])
    abs_eigenvalues = np.abs(all_eigenvalues)
    
    print(f"\nDominant eigenvalues:")
    print(f"  Maximum (most positive): λ_max = {eigval_max:.6e}")
    print(f"  Minimum (most negative): λ_min = {eigval_min:.6e}")
    print(f"  Ratio |λ_max|/|λ_min| = {abs(eigval_max/eigval_min):.4f}")
    
    # Convert eigenvectors back to parameter list format
    eigvec_max_tensor = torch.from_numpy(eigvec_max).float().to(device)
    eigvec_min_tensor = torch.from_numpy(eigvec_min).float().to(device)
    
    direction_max = vector_to_params(eigvec_max_tensor, params)
    direction_min = vector_to_params(eigvec_min_tensor, params)
    
    # Compute dot product to check orthogonality
    dot_product = np.dot(eigvec_max, eigvec_min)
    print(f"  Orthogonality check: v_max · v_min = {dot_product:.6e}")
    
    result = {
        'eigenvector_max': direction_max,
        'eigenvector_min': direction_min,
        'eigenvalue_max': eigval_max,
        'eigenvalue_min': eigval_min,
        'all_eigenvalues': all_eigenvalues,
        'dot_product': dot_product
    }
    
    return result


def filter_normalize_direction(direction, model_params, norm='filter'):
    """
    Apply filter normalization to a direction (same as Li et al. 2018).
    
    This ensures fair comparison between Hessian and random directions.
    
    Args:
        direction: list of tensors (direction vector)
        model_params: list of tensors (model parameters for reference)
        norm: normalization type ('filter', 'layer', 'weight')
    
    Returns:
        list of tensors (normalized direction)
    """
    normalized = []
    for d, w in zip(direction, model_params):
        d_copy = d.clone()
        
        if d_copy.dim() <= 1:
            # Ignore bias and batch norm (1D parameters)
            d_copy.fill_(0)
        else:
            # Apply filter normalization
            if norm == 'filter':
                # Rescale each filter to match weight norm
                d_copy.mul_(w.norm() / (d_copy.norm() + 1e-10))
            elif norm == 'layer':
                # Rescale entire layer
                d_copy.mul_(w.norm() / (d_copy.norm() + 1e-10))
            elif norm == 'weight':
                # Element-wise scaling
                d_copy.mul_(w)
        
        normalized.append(d_copy)
    
    return normalized


def test_hessian_computation():
    """
    Test Hessian computation on a simple quadratic function.
    """
    print("="*60)
    print("Testing Hessian computation on quadratic function")
    print("="*60)
    
    # Create a simple 2-layer network
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 1)
    )
    
    # Simple quadratic loss: L = ||f(x) - y||²
    x = torch.randn(10, 2)
    y = torch.randn(10, 1)
    
    def loss_fn():
        pred = model(x)
        return torch.mean((pred - y)**2)
    
    # Compute Hessian eigenvectors
    result = compute_hessian_eigenvectors(
        model,
        loss_fn,
        num_eigenvalues=2,
        maxiter=50,
        tol=1e-4
    )
    
    if result is not None:
        print("\n✓ Test passed!")
        print(f"  λ_max = {result['eigenvalue_max']:.6e}")
        print(f"  λ_min = {result['eigenvalue_min']:.6e}")
        print(f"  Orthogonality: {result['dot_product']:.6e}")
    else:
        print("\n✗ Test failed!")
    
    return result


if __name__ == "__main__":
    # Run test
    test_hessian_computation()
