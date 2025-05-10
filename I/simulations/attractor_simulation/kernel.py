import numpy as np

def gaussian_kernel(phi, phi_prime, sigma):
    """
    Gaussian coherence filter between configuration vectors φ and φ′.

    Parameters:
    - phi : np.ndarray
        Configuration vector φ = [a, φ, λ, E]
    - phi_prime : np.ndarray
        Previous configuration φ′
    - sigma : np.ndarray
        Gaussian widths for each dimension [σ_a, σ_φ, σ_λ, σ_E]

    Returns:
    - float: Gaussian weighting factor
    """
    delta = phi - phi_prime
    exponent = -0.5 * np.sum((delta / sigma) ** 2)
    return np.exp(exponent)

def entropy_penalty(S_phi, S_phi_prime, lambda_E):
    """
    Penalty factor based on relative entropy between configurations.

    Parameters:
    - S_phi : float
        Entropy at current point
    - S_phi_prime : float
        Entropy at previous point
    - lambda_E : float
        Entanglement penalty coupling

    Returns:
    - float: entropy penalty factor
    """
    S_phi = max(S_phi, 1e-10)
    S_phi_prime = max(S_phi_prime, 1e-10)
    rel_entropy = S_phi * (np.log(S_phi) - np.log(S_phi_prime))
    return np.exp(-lambda_E * rel_entropy)

def K(phi, phi_prime, sigma, S_phi, S_phi_prime, lambda_E):
    """
    Composite transition kernel K(φ, φ′) = coherence filter × entropy penalty.

    Parameters:
    - phi : np.ndarray
        Current configuration vector φ
    - phi_prime : np.ndarray
        Previous configuration vector φ′
    - sigma : np.ndarray
        Gaussian coherence widths
    - S_phi : float
        Entropy at φ
    - S_phi_prime : float
        Entropy at φ′
    - lambda_E : float
        Entanglement coupling parameter

    Returns:
    - float: transition kernel value
    """
    coherence = gaussian_kernel(phi, phi_prime, sigma)
    penalty = entropy_penalty(S_phi, S_phi_prime, lambda_E)
    value = coherence * penalty

    # Safety debug: check for numerical stability
    if not np.isfinite(value):
        print("⚠️ Non-finite K detected")
        print("φ =", phi)
        print("φ′ =", phi_prime)
        print("S(φ) =", S_phi)
        print("S(φ′) =", S_phi_prime)
        print("coherence =", coherence)
        print("penalty =", penalty)

    return value
