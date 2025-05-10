import numpy as np
import time
from kernel import K  # Ensure kernel.py is in the same directory

def initialize_wavefunction(grid_shape, center, width):
    """
    Initialize a normalized 2D Gaussian wavefunction Ψ_0 over (a, φ).
    """
    a = np.linspace(0, 1, grid_shape[0])
    phi = np.linspace(0, 1, grid_shape[1])
    A, P = np.meshgrid(a, phi, indexing='ij')
    grid = np.stack([A, P], axis=-1)

    exponent = -0.5 * (((A - center[0]) / width[0])**2 + ((P - center[1]) / width[1])**2)
    psi = np.exp(exponent)
    psi /= np.sqrt(np.sum(psi**2))
    return psi, grid

def compute_entropy(psi):
    """
    Compute entropy map S_n from ψ_n using −p log p, clipped for safety.
    """
    p = np.clip(psi**2, 1e-12, 1.0)
    return -p * np.log(p)

def recursive_step(psi_n, grid, sigma, lambda_E, verbose=False):
    """
    Compute Ψ_{n+1} from Ψ_n using entropy-aware kernel evolution.
    """
    S_n = compute_entropy(psi_n)
    psi_next = np.zeros_like(psi_n)
    N, M = psi_n.shape

    start_time = time.time()

    for i in range(N):
        for j in range(M):
            phi = grid[i, j]
            E_val = np.sqrt(max(S_n[i, j], 1e-10))
            phi_ext = np.array([phi[0], phi[1], 0.95, E_val])

            val = 0.0
            for k in range(N):
                for l in range(M):
                    phi_prime = grid[k, l]
                    E_prime = np.sqrt(max(S_n[k, l], 1e-10))
                    phi_prime_ext = np.array([phi_prime[0], phi_prime[1], 0.95, E_prime])

                    kernel_val = K(phi_ext, phi_prime_ext, sigma, S_n[i, j], S_n[k, l], lambda_E)
                    val += kernel_val * psi_n[k, l]

            psi_next[i, j] = val

    norm = np.sqrt(np.sum(psi_next**2))
    psi_next = psi_next / norm if norm > 0 else psi_next

    if verbose:
        diff_norm = np.linalg.norm(psi_next - psi_n)
        print(f"Cycle complete in {time.time() - start_time:.2f}s | ‖ΔΨ‖ = {diff_norm:.6f}")

    return psi_next
