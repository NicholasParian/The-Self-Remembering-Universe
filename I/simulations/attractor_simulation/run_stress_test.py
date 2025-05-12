import numpy as np
import os
from evolve import initialize_wavefunction, recursive_step_dynamic

def run_stress_test_grid(
    sigmas = [0.01, 0.02, 0.03, 0.04],
    lambda_E_values = [15.0, 20.0, 25.0, 30.0],

    grid_shape=(30, 30),
    n_steps=50,
    lambda_crit=0.05,
    epsilon=1e-6,
    results_dir="results/stress_test"
):
    os.makedirs(results_dir, exist_ok=True)
    summary = []

    for sigma_val in sigmas:
        for lambda_E in lambda_E_values:
            sigma = [sigma_val] * 4
            label = f"sigma{sigma_val:.2f}_lambdaE{lambda_E:.1f}"
            save_path = os.path.join(results_dir, f"{label}.npz")

            psi_n, grid = initialize_wavefunction(grid_shape, center=[0.5, 0.5], width=[0.1, 0.1])
            psi_prev = psi_n.copy()

            collapsed = False
            for step in range(n_steps):
                psi_next, lambda_n, E_map, collapse_flag = recursive_step_dynamic(
                    psi_n, psi_prev, grid, np.array(sigma), lambda_E, lambda_crit=lambda_crit, verbose=False
                )

                if collapse_flag:
                    collapsed = True
                    break

                if np.linalg.norm(psi_next - psi_n) < epsilon:
                    break

                psi_prev = psi_n.copy()
                psi_n = psi_next.copy()

            np.savez(save_path, collapsed=collapsed, lambda_E=lambda_E, sigma=sigma_val)
            summary.append((sigma_val, lambda_E, collapsed))
            print(f"σ={sigma_val:.2f} | λ_E={lambda_E:.1f} → {'💥 Collapse' if collapsed else '✅ Stable'}")

    return summary

if __name__ == "__main__":
    run_stress_test_grid()
