import numpy as np
from evolve import initialize_wavefunction, recursive_step_dynamic

def run_collapse_trigger_test():
    grid_shape = (20, 20)
    center = [0.5, 0.5]
    width = [0.01, 0.01]  # Sharply peaked wavefunction
    sigma = [0.005] * 4   # Very narrow coherence filter
    lambda_E = 50.0       # Strong entropy penalty
    lambda_crit = 0.1     # Sensitive collapse threshold
    epsilon = 1e-6
    n_steps = 50

    psi_n, grid = initialize_wavefunction(grid_shape, center=center, width=width)
    psi_prev = psi_n.copy()

    print("🚀 Starting engineered collapse test...\n")
    for step in range(n_steps):
        psi_next, lambda_n, E_map, collapse = recursive_step_dynamic(
            psi_n, psi_prev, grid, np.array(sigma), lambda_E, lambda_crit, verbose=True
        )

        delta = np.linalg.norm(psi_next - psi_n)
        print(f"[Step {step+1}] ΔΨ = {delta:.4e}, λₙ = {lambda_n:.6f}, max(Eₙ) = {np.max(E_map):.4f}")

        if collapse:
            print(f"\n💥 Collapse triggered at step {step+1} (λₙ = {lambda_n:.6f})")
            return

        if delta < epsilon:
            print(f"\n✅ Converged at step {step+1} (ΔΨ = {delta:.2e})")
            return

        psi_prev = psi_n
        psi_n = psi_next

    print("\n🟢 No collapse — stable across all steps")

if __name__ == "__main__":
    run_collapse_trigger_test()
