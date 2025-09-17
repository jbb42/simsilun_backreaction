import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
from LTB_model import evolve_LTB

# --- Parameters ---
z_i = 1100
z_f = 0
H_0 = 70
all_ic = True

# --- Labels ---
ROW_LABELS = [f"z = {z_f}", f"z = {z_i}", "Difference"]
COL_LABELS = ["LTB", "Simsilun", "Simsilun (IC)"]


# --- Function to run Simsilun ---
def run_simsilun(z_start, z_end, use_all_ic=True):
    subprocess.run([
        "simsilun/simsilun", str(z_start), str(z_end), str(H_0),
        ".true." if use_all_ic else ".false."
    ], check=True)
    params = np.loadtxt("params")
    rho = np.reshape(params[:, 0], [64, 64])
    theta = np.reshape(params[:, 1], [64, 64])
    sigma = np.reshape(params[:, 2], [64, 64])
    weyl = np.reshape(params[:, 3], [64, 64])
    return rho, theta, sigma, weyl


# --- Evolve LTB ---
rho_i, theta_i, sigma_i, weyl_i, coords = evolve_LTB(0, z_i, z_f, H_0)
rho_f, theta_f, sigma_f, weyl_f, _ = evolve_LTB(1, z_i, z_f, H_0)

# --- Simsilun runs ---
rho_sim_f, theta_sim_f, sigma_sim_f, weyl_sim_f = run_simsilun(z_i, z_f, True)
rho_sim_i, theta_sim_i, sigma_sim_i, weyl_sim_i = run_simsilun(z_i, z_i, True)
rho_simic_f, theta_simic_f, sigma_simic_f, weyl_simic_f = run_simsilun(z_i, z_f, False)
rho_simic_i, theta_simic_i, sigma_simic_i, weyl_simic_i = run_simsilun(z_i, z_i, False)

# --- Compute differences ---
rho_d = rho_f - rho_sim_f
theta_d = theta_f - theta_sim_f
sigma_d = sigma_f - sigma_sim_f
weyl_d = weyl_f - weyl_sim_f

rho_ic_d = rho_f - rho_simic_f
theta_ic_d = theta_f - theta_simic_f
sigma_ic_d = sigma_f - sigma_simic_f
weyl_ic_d = weyl_f - weyl_simic_f

# --- Prepare data grids (rows=z_f,z_i,diff; columns=LTB,Simsilun,Simsilun(IC)) ---
density_grid = [
    [rho_f, rho_sim_f, rho_simic_f],
    [rho_i, rho_sim_i, rho_simic_i],
    [None, rho_d, rho_ic_d]
]

expansion_grid = [
    [theta_f, theta_sim_f, theta_simic_f],
    [theta_i, theta_sim_i, theta_simic_i],
    [None, theta_d, theta_ic_d]
]

shear_grid = [
    [sigma_f, sigma_sim_f, sigma_simic_f],
    [sigma_i, sigma_sim_i, sigma_simic_i],
    [None, sigma_d, sigma_ic_d]
]

weyl_grid = [
    [weyl_f, weyl_sim_f, weyl_simic_f],
    [weyl_i, weyl_sim_i, weyl_simic_i],
    [None, weyl_d, weyl_ic_d]
]


# --- Plotting function with one colorbar per row ---
def plot_grid_row_colorbars(filename, data_grid, coords, cbar_label):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Column titles
    for j, col in enumerate(COL_LABELS):
        axes[0, j].set_title(col, fontsize=14)

    # Row labels
    for i, row in enumerate(ROW_LABELS):
        axes[i, 0].set_ylabel(row, fontsize=14)

    # Plot each panel
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            data = data_grid[i][j]
            if data is not None:
                im = ax.imshow(data, origin='lower',
                               extent=[coords.min(), coords.max(), coords.min(), coords.max()])
            else:
                ax.axis('off')

    # Add one colorbar per row, on the right-hand side
    for i in range(3):
        # Collect all data in this row
        row_data = [data_grid[i][j] for j in range(3) if data_grid[i][j] is not None]
        if len(row_data) == 0:
            continue
        vmin = min(d.min() for d in row_data)
        vmax = max(d.max() for d in row_data)
        # Add invisible im for colorbar
        im = axes[i, 0].imshow(row_data[0], origin='lower',
                               extent=[coords.min(), coords.max(), coords.min(), coords.max()],
                               vmin=vmin, vmax=vmax, alpha=0)
        # attach colorbar to the right of the row (span all columns)
        divider = make_axes_locatable(axes[i, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label=cbar_label)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close(fig)


# --- Plot all ---
plot_grid_row_colorbars("density.pdf", density_grid, coords, r"$\rho / \rho_{\mathrm{EdS}}$")
plot_grid_row_colorbars("expansion.pdf", expansion_grid, coords, r"$\theta / \theta_{\mathrm{EdS}}$")
plot_grid_row_colorbars("shear.pdf", shear_grid, coords, r"$3\sigma / \theta_{\mathrm{EdS}}$")
plot_grid_row_colorbars("weyl.pdf", weyl_grid, coords, r"$\mathcal{W}$")
