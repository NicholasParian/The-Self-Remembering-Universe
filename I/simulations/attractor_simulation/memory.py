import numpy as np

def D(tau, E, beta=1.0, E_max=1.0):
    """
    Decoherence kernel D(τ, E) = γ(E) * exp(-τ / τ_c(E)) * cos(ω(E) * τ)

    Parameters:
    - tau : float or np.ndarray
        Time delay(s) τ
    - E : float
        Entanglement eigenvalue
    - beta : float
        Damping strength (default 1.0)
    - E_max : float
        Maximum entanglement value (default 1.0)

    Returns:
    - np.ndarray
        Kernel values D(τ, E)
    """
    E = max(E, 1e-8)  # Avoid divide-by-zero
    E_max = max(E_max, 1e-8)
    tau_c = 1.0 / E
    gamma = np.exp(-beta / E)
    omega_squared = 1 - (E / E_max)**2
    omega = np.sqrt(max(omega_squared, 0.0))
    return gamma * np.exp(-tau / tau_c) * np.cos(omega * tau)

# --- Optional visualization ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    taus = np.linspace(0, 10, 500)
    E_vals = [0.1, 0.3, 0.6, 0.9]

    plt.figure(figsize=(10, 6))
    for E in E_vals:
        plt.plot(taus, D(taus, E), label=f"E = {E}")

    plt.title("Decoherence Kernel D(τ, E) for Various Entanglement Values")
    plt.xlabel("τ (Delay)")
    plt.ylabel("D(τ, E)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
