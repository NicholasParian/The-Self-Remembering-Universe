import numpy as np
import matplotlib.pyplot as plt
import os

def plot_saved_results(path="results/attractor_run.npz"):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    data = np.load(path)
    norms = data.get("norms")
    fidelities = data.get("fidelities")
    entropies = data.get("entropies")

    if norms is None or fidelities is None or entropies is None:
        print(f"❌ File is missing one or more arrays: norms, fidelities, entropies")
        return

    steps = np.arange(1, len(norms) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(steps, norms, marker='o')
    plt.title("Attractor Convergence")
    plt.xlabel("Cycle n")
    plt.ylabel("‖Ψₙ₊₁ − Ψₙ‖")

    if len(fidelities) > 0:
        plt.subplot(1, 3, 2)
        plt.plot(steps[1:], fidelities, marker='s', color='orange')
        plt.title("Fidelity λₙ")
        plt.xlabel("Cycle n")
        plt.ylabel("|⟨Ψₙ | Ψₙ₋₁⟩|²")

    plt.subplot(1, 3, 3)
    plt.plot(steps[:len(entropies)], entropies, marker='^', color='green')
    plt.title("Recursive Entropy")
    plt.xlabel("Cycle n")
    plt.ylabel("Sₙ")

    plt.tight_layout()
    plt.suptitle(f"Loaded from: {os.path.basename(path)}", y=1.05, fontsize=10)
    plt.show()

# If run directly
if __name__ == "__main__":
    plot_saved_results("results/attractor_run.npz")

