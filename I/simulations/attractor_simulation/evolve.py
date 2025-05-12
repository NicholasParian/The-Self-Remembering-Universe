import numpy as np
from kernel import K
from memory import D

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
    Compute the entropy map S_n using −p log p with numerical safety.
    """
    p = np.clip(psi**2, 1e-12, 1.0)
    return -p * np.log(p)

def compute_fidelity(psi_n, psi_prev):
    """
    Compute inter-cycle fidelity λₙ = |⟨ψₙ | ψₙ₋₁⟩|²
    """
    return float(np.abs(np.sum(psi_n * psi_prev))**2)


def recursive_step(psi_n, grid, sigma, lambda_E, verbose=False):
    """
    Legacy attractor step using fixed λ and E values.
    """
    S_n = compute_entropy(psi_n)
    psi_next = np.zeros_like(psi_n)
    N, M = psi_n.shape

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
    return psi_next / norm if norm > 0 else psi_next

def recursive_step_dynamic(psi_n, psi_prev, grid, sigma, lambda_E, lambda_crit=0.05, verbose=False):
    """
    Evolve Ψ_n to Ψ_{n+1} using dynamic λₙ and Eₙ tracking.

    Returns:
    - psi_next : updated wavefunction
    - lambda_n : computed fidelity
    - E_n_map : entanglement eigenvalue map (√S)
    - collapse_triggered : bool
    """
    S_n = compute_entropy(psi_n)
    psi_next = np.zeros_like(psi_n)
    N, M = psi_n.shape

    lambda_n = compute_fidelity(psi_n, psi_prev)
    E_n_map = np.sqrt(np.clip(S_n, 1e-10, None))

    collapse_triggered = lambda_n < lambda_crit
    if verbose and collapse_triggered:
        print(f"⚠️ Collapse condition met: λ = {lambda_n:.4f} < λ_crit = {lambda_crit:.4f}")

    for i in range(N):
        for j in range(M):
            phi = grid[i, j]
            phi_ext = np.array([phi[0], phi[1], lambda_n, E_n_map[i, j]])

            val = 0.0
            for k in range(N):
                for l in range(M):
                    phi_prime = grid[k, l]
                    phi_prime_ext = np.array([phi_prime[0], phi_prime[1], lambda_n, E_n_map[k, l]])

                    kernel_val = K(phi_ext, phi_prime_ext, sigma, S_n[i, j], S_n[k, l], lambda_E)
                    if i == j == 10:  # middle of grid
                        print(f"→ kernel[center] = {kernel_val:.4e}, φ ≠ φ′ = {not np.allclose(phi_ext, phi_prime_ext)}")

                    val += kernel_val * psi_n[k, l]

            psi_next[i, j] = val

    norm = np.sqrt(np.sum(psi_next**2))
    psi_next = psi_next / norm if norm > 0 else psi_next

    return psi_next, lambda_n, E_n_map, collapse_triggered
    
# Example of a vectorized kernel summation (conceptual)
def recursive_step_vectorized(psi_n, grid, sigma, lambda_E):
    S_n = compute_entropy(psi_n)
    E_n_map = np.sqrt(np.clip(S_n, 1e-10, None))
    lambda_n = 0.95  # or computed dynamically

    N, M = psi_n.shape
    psi_next = np.zeros_like(psi_n)

    # Precompute extended configuration arrays
    phi_ext = np.stack([grid[...,0], grid[...,1], np.full_like(grid[...,0], lambda_n), E_n_map], axis=-1)

    for i in range(N):
        for j in range(M):
            kernel_vals = np.exp(-0.5 * np.sum(((phi_ext[i,j] - phi_ext) / sigma)**2, axis=-1))
            entropy_penalties = np.exp(-lambda_E * S_n[i,j] * (np.log(S_n[i,j]+1e-10) - np.log(S_n+1e-10)))

            psi_next[i,j] = np.sum(kernel_vals * entropy_penalties * psi_n)

    psi_next /= np.sqrt(np.sum(psi_next**2))
    return psi_next


def recursive_step_with_memory(psi_history, E_history, grid, sigma, lambda_E, beta=1.0, kernel_type='oscillatory'):
    """
    Evolve Ψ using full memory integration: Ψ_{n+1} = ∑ D(n−k, E_k) · Ψ_k
    """
    n = len(psi_history) - 1
    N, M = psi_history[0].shape
    psi_next = np.zeros_like(psi_history[0])

    for k in range(n + 1):
        tau = n - k
        D_k = D(tau, E_history[k], beta=beta, kernel_type=kernel_type)
        psi_next += D_k * psi_history[k]  # pointwise weighting

    norm = np.sqrt(np.sum(psi_next**2))
    return psi_next / norm if norm > 0 else psi_next

