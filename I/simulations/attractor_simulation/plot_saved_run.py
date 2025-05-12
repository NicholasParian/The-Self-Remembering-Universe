import numpy as np
import matplotlib.pyplot as plt
import os

def plot_saved_results(path="results/collapse_test_lambdaE10_sigma02.npz", lambda_crit=0.05):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    data = np.load(path)
    norms = data["norms"]
    fidelities = data["fidelities"]
    entropies = data["entropies"]
    E_maxs = data["max_entanglements"]
    collapse_flags = data["collapse_flags"]

    steps = np.arange(1, len(norms) + 1)

    plt.figure(figsize=(16, 6))

    # Convergence norm
    plt.subplot(1, 4, 1)
    plt.plot(steps, norms, marker='o', color='blue')
    plt.title("Attractor Convergence")
    plt.xlabel("Cycle n")
    plt.ylabel("‖Ψₙ₊₁ − Ψₙ‖")
    plt.grid(True, alpha=0.5)

    # Fidelity λₙ
    plt.subplot(1, 4, 2)
    plt.plot(steps, fidelities, marker='s', color='orange')
    plt.axhline(lambda_crit, color='red', linestyle='--', label=f"λₙ₍cᵣᵢₜ₎ = {lambda_crit}")
    plt.title("Fidelity λₙ")
    plt.xlabel("Cycle n")
    plt.ylabel("|⟨Ψₙ | Ψₙ₋₁⟩|²")
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Recursive Entropy Sₙ
    plt.subplot(1, 4, 3)
    plt.plot(steps, entropies, marker='^', color='green')
    plt.title("Recursive Entropy Sₙ")
    plt.xlabel("Cycle n")
    plt.ylabel("Entropy")
    plt.grid(True, alpha=0.5)

    # Max Entanglement Eₙ
    plt.subplot(1, 4, 4)
    plt.plot(steps, E_maxs, marker='x', color='purple')
    plt.title("Max Entanglement Eₙ")
    plt.xlabel("Cycle n")
    plt.ylabel("max(Eₙ)")
    plt.grid(True, alpha=0.5)

    # Highlight collapse if detected
    collapse_indices = np.where(collapse_flags)[0]
    if collapse_indices.size > 0:
        collapse_index = collapse_indices[0] + 1
        for ax in plt.gcf().axes:
            ax.axvline(collapse_index, linestyle='--', color='red', alpha=0.6)
        plt.suptitle(f"Collapse Detected at Step {collapse_index}", fontsize=16, color='red')
    else:
        plt.suptitle("No Collapse Detected", fontsize=16, color='green')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_saved_results("results/collapse_test_lambdaE10_sigma02.npz")
