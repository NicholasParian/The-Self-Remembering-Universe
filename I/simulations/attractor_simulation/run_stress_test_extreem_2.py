import numpy as np
from evolve import initialize_wavefunction, recursive_step_dynamic

def run_collapse_trigger_test(
    grid_shape=(20, 20),
    width=[0.01, 0.01],
    sigma=[0.005, 0.005, 0.005, 0.005],
    lambda_E=50.0,
    lambda_crit=0.05,
    epsilon=1e-6,
    n_steps=50
):
    print("🔬 Running collapse trigger test...")

    psi_n, grid = initialize_wavefunction(grid_shape, center=[0.5, 0.5], width=width)
    psi_prev = psi_n.copy()

    for step in range(n_steps):
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
    run_collapse_trigger_test()
