import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if available
import matplotlib.pyplot as plt
import numpy as np

def plot_grid(g_size):
    initial = np.load("./data/jusilun_output/initial_vals_043.npz")
    final = np.load("./data/jusilun_output/final_vals_043.npz")
    rho_i = initial["rho"]
    rho_i_bg = rho_i[-1]
    rho_i = rho_i[:-1].reshape(g_size,g_size,g_size)
    rho_f = final["rho"]
    rho_f_bg = rho_f[-1]
    rho_f = rho_f[:-1].reshape(g_size,g_size,g_size)

    plt.figure()
    plt.imshow(rho_i[:,:,g_size//2]/rho_i_bg)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(rho_f[:,:,g_size//2]/rho_f_bg)
    plt.colorbar()
    plt.show()
    return
plot_grid(128)