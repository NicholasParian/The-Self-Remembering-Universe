import numpy as np
import matplotlib.pyplot as plt

# === Kernel functions ===
def gaussian_kernel(phi, phi_prime, sigma):
    delta = phi - phi_prime
    z = delta / np.maximum(sigma, 1e-6)
    exponent = -0.5 * np.sum(z ** 2)
    if exponent < -100:
        return 0.0
    return np.exp(exponent)

def entropy_penalty(S_phi, S_phi_prime, lambda_E):
    S_phi = max(S_phi, 1e-10)
    S_phi_prime = max(S_phi_prime, 1e-10)
    rel_entropy = S_phi * (np.log(S_phi) - np.log(S_phi_prime))
    return np.exp(-lambda_E * rel_entropy)

def K(phi, phi_prime, sigma, S_phi, S_phi_prime, lambda_E):
    coherence = gaussian_kernel(phi, phi_prime, sigma)
    penalty = entropy_penalty(S_phi, S_phi_prime, lambda_E)
    return coherence * penalty

# === Memory kernel (not used here directly) ===
def D(tau, E, beta=1.0, E_max=1.0, kernel_type='oscillatory'):
    E = np.clip(E, 1e-8, None)
    if kernel_type == 'oscillatory':
        E_max = max(E_max, 1e-8)
        tau_c = 1.0 / E
        gamma = np.exp(-beta / E)
        omega_squared = 1 - (E / E_max)**2
        omega = np.sqrt(np.maximum(omega_squared, 0.0))
        return gamma * np.exp(-tau / tau_c) * np.cos(omega * tau)
    elif kernel_type == 'gaussian':
        tau_c = 1.0 / E
        return np.exp(- (tau / tau_c)**2)
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

# === Evolution functions ===
def initialize_wavefunction(grid_shape, center, width):
    a = np.linspace(0, 1, grid_shape[0])
    phi = np.linspace(0, 1, grid_shape[1])
    A, P = np.meshgrid(a, phi, indexing='ij')
    exponent = -0.5 * (((A - center[0]) / width[0])**2 + ((P - center[1]) / width[1])**2)
    psi = np.exp(exponent)
    psi /= np.sqrt(np.sum(psi**2))
    grid = np.stack([A, P], axis=-1)
    return psi, grid

def compute_entropy(psi):
    p = np.clip(psi**2, 1e-12, 1.0)
    return -p * np.log(p)

def compute_fidelity(psi_n, psi_prev):
    return float(np.abs(np.sum(psi_n * psi_prev))**2)

def recursive_step_dynamic(psi_n, psi_prev, grid, sigma, lambda_E, lambda_crit=0.05):
    S_n = compute_entropy(psi_n)
    psi_next = np.zeros_like(psi_n)
    N, M = psi_n.shape

    lambda_n = compute_fidelity(psi_n, psi_prev)
    E_n_map = np.sqrt(np.clip(S_n, 1e-10, None))
    collapse_triggered = lambda_n < lambda_crit

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
                    val += kernel_val * psi_n[k, l]

            psi_next[i, j] = val

    norm = np.sqrt(np.sum(psi_next**2))
    psi_next = psi_next / norm if norm > 0 else psi_next

    return psi_next, lambda_n, E_n_map, collapse_triggered

# === Parameter sweep ===
def run_simulation_for_params(grid_shape, sigma, lambda_E, lambda_crit, epsilon, n_steps):
    psi_n, grid = initialize_wavefunction(grid_shape, center=[0.5, 0.5], width=[0.05, 0.05])
    psi_prev = psi_n.copy()

    for step in range(n_steps):
        psi_next, lambda_n, E_map, collapse_triggered = recursive_step_dynamic(
            psi_n, psi_prev, grid, sigma, lambda_E, lambda_crit
        )

        diff_norm = np.linalg.norm(psi_next - psi_n)
        if collapse_triggered:
            return "collapse"
        elif diff_norm < epsilon:
            return "converged"

        psi_prev = psi_n
        psi_n = psi_next

    return "incomplete"

# === Run parameter grid ===
sigma_vals = np.linspace(0.01, 0.2, 10)
lambda_E_vals = np.linspace(0.1, 10.0, 10)
results = np.empty((len(sigma_vals), len(lambda_E_vals)), dtype=object)

for i, sigma_val in enumerate(sigma_vals):
    for j, lambda_E_val in enumerate(lambda_E_vals):
        sigma = np.array([sigma_val] * 4)
        result = run_simulation_for_params(
            grid_shape=(20, 20),
            sigma=sigma,
            lambda_E=lambda_E_val,
            lambda_crit=0.05,
            epsilon=1e-6,
            n_steps=50
        )
        results[i, j] = result
        print(f"σ={sigma_val:.3f} | λ_E={lambda_E_val:.2f} → {result}")

# === Plot ===
status_map = {"collapse": 0, "converged": 1, "incomplete": 2}
int_results = np.vectorize(lambda x: status_map[x])(results)

plt.figure(figsize=(8, 6))
cmap = plt.cm.get_cmap("RdYlGn", 3)
im = plt.imshow(int_results, origin='lower', cmap=cmap, extent=[0.1, 10.0, 0.01, 0.2], aspect='auto')
plt.colorbar(im, ticks=[0, 1, 2], label='Status')
plt.clim(-0.5, 2.5)
plt.xticks(np.round(lambda_E_vals, 1), rotation=45)
plt.yticks(np.round(sigma_vals, 2))
plt.xlabel("λ_E (Entanglement Penalty Strength)")
plt.ylabel("σ (Gaussian Width)")
plt.title("Convergence vs Collapse Zone Map")
plt.tight_layout()
plt.show()
