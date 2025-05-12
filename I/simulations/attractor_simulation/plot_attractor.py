import numpy as np
import matplotlib.pyplot as plt
import time
import os
from evolve import initialize_wavefunction, recursive_step_dynamic, recursive_step_with_memory

def run_simulation(
    n_steps=50,
    grid_shape=(30, 30),
    sigma=[0.2, 0.2, 0.2, 0.2],
    lambda_E=10.0,
    epsilon=1e-6,
    lambda_crit=0.05,
    verbose=True,
    save_path="results/collapse_test_lambdaE10_sigma02.npz"
):
    center, width = np.array([0.5, 0.5]), np.array([0.1, 0.1])
    psi_n, grid = initialize_wavefunction(grid_shape, center, width)
    psi_prev = psi_n.copy()

    norms, fidelities, entropies, max_entanglements, collapse_flags = [], [], [], [], []

    for step in range(n_steps):
        start_time = time.time()
        
        psi_next = recursive_step_with_memory(psi_history, E_history, grid, np.array(sigma), lambda_E)

        lambda_n = compute_fidelity(psi_next, psi_n)
        E_map = np.sqrt(np.clip(compute_entropy(psi_next), 1e-10, None))
        collapse_triggered = lambda_n < lambda_crit

        diff_norm = np.linalg.norm(psi_next - psi_n)
        norms.append(diff_norm)
        fidelities.append(lambda_n)
        entropy = -np.sum(np.clip(psi_next**2, 1e-12, 1.0) * np.log(np.clip(psi_next**2, 1e-12, 1.0)))
        entropies.append(entropy)
        max_E = np.max(E_map)
        max_entanglements.append(max_E)
        collapse_flags.append(collapsed)

        elapsed = time.time() - start_time
        if verbose:
            print(f"[Cycle {step+1}/{n_steps}] ΔΨ={diff_norm:.2e}, λ={lambda_n:.4f}, "
                  f"E_max={max_E:.4f}, Entropy={entropy:.4f}, Time={elapsed:.2f}s")

        if collapsed:
            print(f"❗ Collapse triggered at step {step+1} (λ={lambda_n:.4f})")
            break

        if diff_norm < epsilon:
            print(f"✅ Converged at step {step+1} (ΔΨ={diff_norm:.2e})")
            break

        psi_prev = psi_n.copy()
        psi_n = psi_next.copy()

    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        norms=norms,
        fidelities=fidelities,
        entropies=entropies,
        max_entanglements=max_entanglements,
        collapse_flags=collapse_flags,
    )
    print(f"📦 Results saved to {save_path}")

    return norms, fidelities, entropies, max_entanglements, collapse_flags

# --- Run and Plot ---
if __name__ == "__main__":
    norms, fidelities, entropies, E_maxs, collapses = run_simulation()
    steps = np.arange(1, len(norms) + 1)

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, 1)
    plt.plot(steps, norms, marker='o')
    plt.title("Attractor Convergence")
    plt.xlabel("Cycle n")
    plt.ylabel("‖Ψₙ₊₁ − Ψₙ‖")

    plt.subplot(1, 4, 2)
    plt.plot(steps, fidelities, marker='s', color='orange')
    plt.axhline(0.05, color='red', linestyle='--', label='Collapse Threshold')
    plt.title("Fidelity λₙ")
    plt.xlabel("Cycle n")
    plt.ylabel("|⟨Ψₙ | Ψₙ₋₁⟩|²")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(steps, entropies, marker='^', color='green')
    plt.title("Recursive Entropy Sₙ")
    plt.xlabel("Cycle n")
    plt.ylabel("Entropy")

    plt.subplot(1, 4, 4)
    plt.plot(steps, E_maxs, marker='x', color='purple')
    plt.title("Max Entanglement Eₙ")
    plt.xlabel("Cycle n")
    plt.ylabel("max(Eₙ)")

    plt.tight_layout()
    plt.show()
