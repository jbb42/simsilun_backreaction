import numpy as np
import subprocess
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from LTB_model import evolve_LTB

# --- Parameters ---
z_i, z_f, H_0 = 1100, 0, 70
g_size = 64
coords = (np.arange(g_size) - (g_size - 1) / 2)

ROW_LABELS = ["z = " + str(z_i), "z = " + str(z_f), "Difference"]
COL_LABELS = ["LTB", "Simsilun", "Simsilun (IC)"]


# --- Helper functions ---
def run_simsilun(z_start, z_end, use_all_ic=True):
    subprocess.run([
        "simsilun/simsilun", str(z_start), str(z_end), str(H_0),
        ".true." if use_all_ic else ".false."
    ], check=True)
    p = np.loadtxt("params").reshape(-1, 5)
    return [p[:, i].reshape(g_size, g_size, g_size) for i in range(5)]  # rho, theta, sigma, weyl, V


# --- Slice helper ---
def get_slice(data, axis=2, index=None):
    if index is None:
        index = data.shape[axis] // 2  # middle slice
    if axis == 0:
        return data[index, :, :]
    elif axis == 1:
        return data[:, index, :]
    elif axis == 2:
        return data[:, :, index]


# --- Modified plot_grid ---
def plot_grid(name, grid, coords, cbar_label, slice_axis=2, slice_index=None):
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # Column titles
    for j, col in enumerate(COL_LABELS):
        axes[0, j].set_title(col, fontsize=14)

    for i in range(3):
        row_data = [get_slice(d, slice_axis, slice_index) for d in grid[i] if d is not None]

        if i == 2:  # Difference row: one colorbar per plot
            for j, d in enumerate(grid[i]):
                ax = axes[i, j]
                if d is None:
                    ax.axis('off')
                    continue
                d2d = get_slice(d, slice_axis, slice_index)
                im = ax.imshow(d2d, origin='lower', extent=[coords.min(), coords.max()] * 2)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="6%", pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical', label=cbar_label)
        else:  # z rows: one shared colorbar for the row
            vmin, vmax = min(d.min() for d in row_data), max(d.max() for d in row_data)
            for j, d in enumerate(grid[i]):
                ax = axes[i, j]
                if d is None:
                    ax.axis('off')
                    continue
                d2d = get_slice(d, slice_axis, slice_index)
                ax.imshow(d2d, origin='lower', extent=[coords.min(), coords.max()] * 2,
                          vmin=vmin, vmax=vmax)
            # Shared colorbar using ScalarMappable
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=plt.cm.viridis)
            sm.set_array([])
            divider = make_axes_locatable(axes[i, -1])
            cax = divider.append_axes("right", size="6%", pad=0.05)
            fig.colorbar(sm, cax=cax, orientation='vertical', label=cbar_label)

        # Row labels (first visible axis in row)
        left_ax = next((ax for ax in axes[i, :] if ax.get_visible()), None)
        if left_ax:
            left_ax.set_ylabel(ROW_LABELS[i], fontsize=14)

    fig.text(0.30, 0.17, "Difference", ha='center', va='center', rotation=90, fontsize=14)
    fig.text(0.10, 0.10, name, ha='center', va='center', rotation=0, fontsize=28)
    plt.tight_layout()
    plt.savefig(f"{name}.pdf")
    plt.show()
    plt.close(fig)



# --- Evolve LTB ---
rho_i, theta_i, sigma_i, weyl_i, V_i = evolve_LTB(0, z_i, z_f, H_0)
#rho_i = np.load("data/grid_rho0.npy")
#theta_i = np.load("data/grid_theta0.npy")
#sigma_i = np.load("data/grid_sigma0.npy")
#weyl_i = np.load("data/grid_weyl0.npy")
#V_i = np.load("data/grid_V0.npy")

np.savetxt("rho_i", rho_i.reshape(g_size**3))
np.savetxt("theta_i", theta_i.reshape(g_size**3))
np.savetxt("sigma_i", sigma_i.reshape(g_size**3))
np.savetxt("weyl_i", weyl_i.reshape(g_size**3))
np.savetxt("V_i", V_i.reshape(g_size**3))

rho_f, theta_f, sigma_f, weyl_f, V_f = evolve_LTB(1, z_i, z_f, H_0)
#rho_f = np.load("data/grid_rho1.npy")
#theta_f = np.load("data/grid_theta1.npy")
#sigma_f = np.load("data/grid_sigma1.npy")
#weyl_f = np.load("data/grid_weyl1.npy")
#V_f = np.load("data/grid_V1.npy")

# --- Simsilun runs ---
runs = {
    "sim_f": run_simsilun(z_i, z_f, True),
    "sim_i": run_simsilun(z_i, z_i, True),
    "simic_f": run_simsilun(z_i, z_f, False),
    "simic_i": run_simsilun(z_i, z_i, False)
}

# --- Quantities to plot ---
quantities = {
    "Density": {"LTB": (rho_f, rho_i), "Simsilun": (runs["sim_f"][0], runs["sim_i"][0]),
                "Simsilun(IC)": (runs["simic_f"][0], runs["simic_i"][0])},
    "Expansion": {"LTB": (theta_f, theta_i), "Simsilun": (runs["sim_f"][1], runs["sim_i"][1]),
                  "Simsilun(IC)": (runs["simic_f"][1], runs["simic_i"][1])},
    "Shear": {"LTB": (sigma_f, sigma_i), "Simsilun": (runs["sim_f"][2], runs["sim_i"][2]),
              "Simsilun(IC)": (runs["simic_f"][2], runs["simic_i"][2])},
    "Weyl": {"LTB": (weyl_f, weyl_i), "Simsilun": (runs["sim_f"][3], runs["sim_i"][3]),
             "Simsilun(IC)": (runs["simic_f"][3], runs["simic_i"][3])},
    "Volume": {"LTB": (V_f, V_i), "Simsilun": (runs["sim_f"][4], runs["sim_i"][4]),
             "Simsilun(IC)": (runs["simic_f"][4], runs["simic_i"][4])},
}

labels = {
    "Density": r"$\rho / \rho_{\mathrm{EdS}}$",
    "Expansion": r"$\theta / \theta_{\mathrm{EdS}}$",
    "Shear": r"$3\sigma / \theta_{\mathrm{EdS}}$",
    "Weyl": r"$\mathcal{W}$",
    "Volume": r"$V$"# / V_{\mathrm{EdS}}$"
}

# --- Plotting ---
for name, q in quantities.items():
    # Compute differences
    diff1 = q["LTB"][0] - q["Simsilun"][0]
    diff2 = q["LTB"][0] - q["Simsilun(IC)"][0]

    # Build grid with difference row centered (first column empty)
    grid = [
        [q["LTB"][1], q["Simsilun"][1], q["Simsilun(IC)"][1]],
        [q["LTB"][0], q["Simsilun"][0], q["Simsilun(IC)"][0]],
        [None, diff1, diff2]
    ]
    plot_grid(name, grid, coords, labels[name])
