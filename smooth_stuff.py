import numpy as np


def gm(X, V):
    """Volume-weighted mean."""
    return np.sum(X * V) / np.sum(V)


def compare_Omega_Q(filename):
    data = np.load(filename)

    theta = data["theta"].astype(np.float64)
    V = data["V"].astype(np.float64)

    # Check if sigma in file is already squared or scalar
    sigma = data["sigma"].astype(np.float64)  # assume sigma = sqrt(sigma²)
    sigma2 = sigma ** 2

    # Volume-weighted H
    H = gm(theta, V) / 3

    # Direct method
    theta2_mean = gm(theta ** 2, V)
    sigma2_mean = gm(sigma2, V)
    Omega_Q_direct = (gm(sigma ** 2, V) + 1 / 9 * gm(theta ** 2, V) - H ** 2) / H ** 2

    # Q-first method
    Q = (2 / 3) * (theta2_mean - gm(theta, V) ** 2) - 2 * sigma2_mean
    Omega_Q_from_Q = - Q / (6 * H ** 2)

    # Compare
    abs_diff = np.abs(Omega_Q_direct - Omega_Q_from_Q)
    rel_diff = abs_diff / np.abs(Omega_Q_direct) if Omega_Q_direct != 0 else np.nan

    print(f"File: {filename}")
    print(f"Omega_Q (direct)  = {Omega_Q_direct:.10f}")
    print(f"Omega_Q (from Q)  = {Omega_Q_from_Q:.10f}")
    print(f"Absolute diff     = {abs_diff:.10e}")
    print(f"Relative diff     = {rel_diff:.10e}\n")


# Example usage for multiple files
for i in range(45):
    filename = f"./data/jusilun_output/final_vals_{str(i).zfill(3)}.npz"
    compare_Omega_Q(filename)



