import numpy as np
from evolve import initialize_wavefunction, recursive_step_dynamic, compute_entropy, compute_fidelity

def run_collapse_trigger_test_force_collapse(
    grid_shape=(30, 30),
    width = [0.15, 0.15],
    sigma = [0.1, 0.1, 0.1, 0.1],
    lambda_E=10.0,
    lambda_crit=0.99,
    epsilon=1e-6,
    n_steps=50,
    noise_level=0.01
):
    print("🔬 Running forced collapse test...")

    psi_n, grid = initialize_wavefunction(grid_shape, center=[0.5, 0.5], width=width)
    psi_prev = psi_n.copy()

    for step in range(n_steps):
        # Add small noise to simulate decoherence events
        psi_n += noise_level * np.random.randn(*psi_n.shape)
        psi_n /= np.linalg.norm(psi_n)

        psi_next, lambda_n, E_map, collapse_flag = recursive_step_dynamic(
            psi_n, psi_prev, grid, np.array(sigma), lambda_E, lambda_crit=lambda_crit, verbose=True
        )

        diff = np.linalg.norm(psi_next - psi_n)
        print(f"Step {step+1} | λₙ = {lambda_n:.6f} | ‖ΔΨ‖ = {diff:.2e} | Eₘₐₓ = {np.max(E_map):.4f}")

        if collapse_flag:
            print(f"❗ Collapse triggered at step {step+1} (λₙ = {lambda_n:.6f})")
            return

        if diff < epsilon:
            print(f"✅ Converged at step {step+1} (‖ΔΨ‖ = {diff:.2e})")
            return

        psi_prev = psi_n
        psi_n = psi_next

    print("⚠️ Simulation completed without collapse or convergence.")

if __name__ == "__main__":
    run_collapse_trigger_test_force_collapse()
