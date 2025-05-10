import numpy as np
import matplotlib.pyplot as plt
import time
import os
from evolve import initialize_wavefunction, recursive_step

def compute_fidelity(psi_n, psi_prev):
    """
    Compute fidelity λₙ = |⟨ψₙ | ψₙ₋₁⟩|²
    """
    return np.abs(np.sum(psi_n * psi_prev))**2

def run_simulation(
    n_steps=150,
    grid_shape=(40, 40),
    sigma=[0.1, 0.1, 0.05, 0.05],
    lambda_E=5.0,
    epsilon=1e-6,
    verbose=True,
    save_path="results/long_run_40x40_sigma005_lambda5.npz"
):
    """
    Simulates recursive evolution and saves results.

    Parameters:
    - n_steps : int
    - grid_shape : tuple
    - sigma : list of floats
    - lambda_E : float
    - epsilon : float
    - verbose : bool
    - save_path : str
    """
    center = np.array([0.5, 0.5])
    width = np.array([0.1, 0.1])
    psi_n, grid = initialize_wavefunction(grid_shape, center, width)

    norms, fidelities, entropies = [], [], []

    for step in range(n_steps):
        start = time.time()
        psi_next = recursive_step(psi_n, grid, np.array(sigma), lambda_E, verbose=False)

        diff_norm = np.linalg.norm(psi_next - psi_n)
        norms.append(diff_norm)

        if step > 0:
            fidelities.append(compute_fidelity(psi_next, psi_n))

        p = np.clip(psi_next**2, 1e-12, 1.0)
        entropy = -np.sum(p * np.log(p))
        entropies.append(entropy)

        if verbose:
            print(f"Cycle {step+1}/{n_steps} ‖ΔΨ‖ = {diff_norm:.4e} | Sₙ = {entropy:.4f} | Time = {time.time() - start:.2f}s")

        if diff_norm < epsilon:
            if verbose:
                print(f"✅ Converged at step {step+1} with ‖ΔΨ‖ = {diff_norm:.2e}")
            break

        psi_n = psi_next.copy()

    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, norms=norms, fidelities=fidelities, entropies=entropies)
    if verbose:
        print(f"📦 Results saved to {save_path}")

    return norms, fidelities, entropies

# --- Run and Plot ---
norms, fidelities, entropies = run_simulation()
steps = np.arange(1, len(norms) + 1)

plt.figure(figsize=(12, 4))

# Convergence norm
plt.subplot(1, 3, 1)
plt.plot(steps, norms, marker='o')
plt.title("Attractor Convergence")
plt.xlabel("Cycle n")
plt.ylabel("‖Ψₙ₊₁ − Ψₙ‖")

# Fidelity
if fidelities:
    plt.subplot(1, 3, 2)
    plt.plot(steps[1:], fidelities, marker='s', color='orange')
    plt.title("Fidelity λₙ")
    plt.xlabel("Cycle n")
    plt.ylabel("|⟨Ψₙ | Ψₙ₋₁⟩|²")

# Entropy
plt.subplot(1, 3, 3)
plt.plot(steps, entropies, marker='^', color='green')
plt.title("Recursive Entropy")
plt.xlabel("Cycle n")
plt.ylabel("Sₙ")

plt.tight_layout()
plt.show()
