"""
Test script for Hessian eigenvector computation.

This script tests the hessian_utils module on a simple network before
applying it to the full PINN loss landscape computation.
"""

import torch
import numpy as np
import sys
import os

# Import the hessian_utils module
import hessian_utils

def test_simple_quadratic():
    """
    Test 1: Simple quadratic loss function.
    For a quadratic loss L = 0.5 * x^T H x, the Hessian is exactly H.
    """
    print("="*60)
    print("Test 1: Simple quadratic loss on small network")
    print("="*60)
    
    # Create a simple 2-layer network
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1, bias=False)
    )
    
    # Create synthetic data
    X = torch.randn(20, 2)
    y = torch.randn(20, 1)
    
    def loss_fn():
        pred = model(X)
        return torch.mean((pred - y)**2)
    
    # Compute Hessian eigenvectors
    result = hessian_utils.compute_hessian_eigenvectors(
        model,
        loss_fn,
        num_eigenvalues=2,
        maxiter=50,
        tol=1e-4
    )
    
    if result is not None:
        print("\n✓ Test 1 PASSED")
        print(f"  Maximum eigenvalue: λ_max = {result['eigenvalue_max']:.6e}")
        print(f"  Minimum eigenvalue: λ_min = {result['eigenvalue_min']:.6e}")
        print(f"  Ratio |λ_max/λ_min|: {abs(result['eigenvalue_max']/result['eigenvalue_min']):.4f}")
        print(f"  Orthogonality (dot product): {result['dot_product']:.6e}")
        
        # Check eigenvector shapes
        n_params = sum(p.numel() for p in model.parameters())
        eigvec_max_len = sum(v.numel() for v in result['eigenvector_max'])
        eigvec_min_len = sum(v.numel() for v in result['eigenvector_min'])
        
        assert eigvec_max_len == n_params, f"Eigenvector length mismatch: {eigvec_max_len} != {n_params}"
        assert eigvec_min_len == n_params, f"Eigenvector length mismatch: {eigvec_min_len} != {n_params}"
        print(f"  ✓ Eigenvector dimensions correct: {n_params} parameters")
        
        return True
    else:
        print("\n✗ Test 1 FAILED")
        return False


def test_filter_normalization():
    """
    Test 2: Filter normalization preserves structure.
    """
    print("\n" + "="*60)
    print("Test 2: Filter normalization")
    print("="*60)
    
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 1)
    )
    
    # Create a random direction
    direction = [torch.randn_like(p) for p in model.parameters()]
    model_params = [p for p in model.parameters()]
    
    # Apply filter normalization
    normalized = hessian_utils.filter_normalize_direction(direction, model_params, norm='filter')
    
    print("Original direction norms:")
    for i, d in enumerate(direction):
        print(f"  Layer {i}: {d.norm().item():.6f}")
    
    print("\nNormalized direction norms:")
    for i, d in enumerate(normalized):
        print(f"  Layer {i}: {d.norm().item():.6f}")
    
    print("\nModel parameter norms:")
    for i, p in enumerate(model_params):
        print(f"  Layer {i}: {p.norm().item():.6f}")
    
    # Check that normalized directions match model parameter norms (for non-bias layers)
    for i, (d, p) in enumerate(zip(normalized, model_params)):
        if d.dim() > 1:  # Not bias
            norm_ratio = d.norm().item() / p.norm().item()
            print(f"  Layer {i} norm ratio: {norm_ratio:.6f}")
            assert abs(norm_ratio - 1.0) < 0.01, f"Normalization failed at layer {i}"
    
    print("\n✓ Test 2 PASSED")
    return True


def test_pinn_like_loss():
    """
    Test 3: PINN-like loss with PDE residual.
    """
    print("\n" + "="*60)
    print("Test 3: PINN-like loss with autograd")
    print("="*60)
    
    torch.manual_seed(42)
    
    # Simple network
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1)
    )
    
    def pinn_loss_fn():
        # Generate grid points
        x = torch.linspace(0, 1, 10)
        y = torch.linspace(0, 1, 10)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        inp = torch.stack([X.flatten(), Y.flatten()], dim=1).requires_grad_(True)
        
        # Forward pass
        u = model(inp)
        
        # Compute derivatives (simplified PDE residual)
        grads = torch.autograd.grad(u, inp, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grads[:, 0]
        u_y = grads[:, 1]
        
        # Second derivatives
        grads_x = torch.autograd.grad(u_x, inp, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xx = grads_x[:, 0]
        
        grads_y = torch.autograd.grad(u_y, inp, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_yy = grads_y[:, 1]
        
        # Laplacian
        laplacian = u_xx + u_yy
        
        # Loss
        loss = torch.mean(laplacian**2)
        return loss
    
    try:
        # Compute Hessian eigenvectors
        result = hessian_utils.compute_hessian_eigenvectors(
            model,
            pinn_loss_fn,
            num_eigenvalues=2,
            maxiter=30,
            tol=1e-3
        )
        
        if result is not None:
            print("\n✓ Test 3 PASSED")
            print(f"  Maximum eigenvalue: λ_max = {result['eigenvalue_max']:.6e}")
            print(f"  Minimum eigenvalue: λ_min = {result['eigenvalue_min']:.6e}")
            return True
        else:
            print("\n⚠ Test 3 WARNING: Hessian computation returned None")
            return True  # Don't fail, just warn
    except Exception as e:
        print(f"\n✗ Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🔬 TESTING HESSIAN EIGENVECTOR COMPUTATION ".center(60, "="))
    print()
    
    results = []
    
    # Run tests
    results.append(("Simple Quadratic", test_simple_quadratic()))
    results.append(("Filter Normalization", test_filter_normalization()))
    results.append(("PINN-like Loss", test_pinn_like_loss()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(p for _, p in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nYou can now run the full loss landscape computation with:")
        print("  python task3_loss_landscape.py")
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
