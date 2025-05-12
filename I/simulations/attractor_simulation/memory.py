import numpy as np

def D(tau, E, beta=1.0, E_max=1.0, kernel_type='oscillatory'):
    """
    Decoherence kernel D(τ, E).

    Parameters:
    - tau : float or np.ndarray
        Time delay(s) τ.
    - E : float
        Entanglement eigenvalue.
    - beta : float, optional
        Damping strength (default: 1.0).
    - E_max : float, optional
        Maximum entanglement eigenvalue (default: 1.0).
    - kernel_type : str, optional
        Kernel functional form ('oscillatory' or 'gaussian').

    Returns:
    - np.ndarray
        Kernel values D(τ, E).
    """
    E = np.maximum(E, 1e-8)  # Elementwise safeguard


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

# --- Optional visualization ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    taus = np.linspace(0, 10, 500)
    E_vals = [0.1, 0.3, 0.6, 0.9]

    kernel_types = ['oscillatory', 'gaussian']

    for kernel in kernel_types:
        plt.figure(figsize=(10, 6))
        for E in E_vals:
            plt.plot(taus, D(taus, E, kernel_type=kernel), label=f"E = {E}")

        plt.title(f"Decoherence Kernel ({kernel.capitalize()}), D(τ, E)")
        plt.xlabel("τ (Delay)")
        plt.ylabel("D(τ, E)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
