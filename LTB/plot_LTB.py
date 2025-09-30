import numpy as np
import subprocess
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from LTB.LTB_model import evolve_LTB

# --- Parameters ---
z_i, z_f, H_0 = 1100, 0, 70
usefiles=True
g_size = 32
coords = (np.arange(g_size) - (g_size - 1) / 2)

ROW_LABELS = ["z = " + str(z_i), "z = " + str(z_f), "Difference"]
COL_LABELS = ["LTB", "Simsilun", "Simsilun (IC)"]


# --- Helper functions ---
def run_simsilun(z_start, z_end, use_all_ic=True):
    subprocess.run([
        "../simsilun/simsilun", str(z_start), str(z_end), str(H_0),
        ".true." if use_all_ic else ".false."
    ], check=True)
    p = np.loadtxt("../data/simsilun_output/params").reshape(-1, 5)
    np.save(f"../data/np_arrays_LTB/params_z{z_end}_all_ic_{use_all_ic}.npy", p)
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
            vmin = min(np.nanmin(d) for d in row_data)
            vmax = max(np.nanmax(d) for d in row_data)
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
    plt.savefig(f"../plots/{name}.pdf")
    plt.show()
    plt.close(fig)



# --- Evolve LTB ---
if not usefiles:
    rho_i, theta_i, sigma_i, weyl_i, V_i = evolve_LTB(0, z_i, z_f, H_0)
    rho_f, theta_f, sigma_f, weyl_f, V_f = evolve_LTB(1, z_i, z_f, H_0)
else:
    rho_i = np.load("../data/np_arrays_LTB/grid_rho0.npy")
    theta_i = np.load("../data/np_arrays_LTB/grid_theta0.npy")
    sigma_i = np.load("../data/np_arrays_LTB/grid_sigma0.npy")
    weyl_i = np.load("../data/np_arrays_LTB/grid_weyl0.npy")
    V_i = np.load("../data/np_arrays_LTB/grid_V0.npy")

    rho_f = np.load("../data/np_arrays_LTB/grid_rho1.npy")
    theta_f = np.load("../data/np_arrays_LTB/grid_theta1.npy")
    sigma_f = np.load("../data/np_arrays_LTB/grid_sigma1.npy")
    weyl_f = np.load("../data/np_arrays_LTB/grid_weyl1.npy")
    V_f = np.load("../data/np_arrays_LTB/grid_V1.npy")

np.savetxt("../data/ics/rho_i", rho_i.reshape(g_size ** 3))
np.savetxt("../data/ics/theta_i", theta_i.reshape(g_size ** 3))
np.savetxt("../data/ics/sigma_i", sigma_i.reshape(g_size ** 3))
np.savetxt("../data/ics/weyl_i", weyl_i.reshape(g_size ** 3))
np.savetxt("../data/ics/V_i", V_i.reshape(g_size ** 3))



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


def compute_Q_grid(theta, sigma, V, center, rad, rmax=32):
    # --- coordinate-based radial cumulative sum ---
    def radial_cumsum(arr, center):
        zc, yc, xc = center
        z, y, x = np.indices(arr.shape)
        r = np.sqrt((z - zc)**2 + (y - yc)**2 + (x - xc)**2)
        flat_arr, flat_r = arr.ravel(), r.ravel()
        idx = np.argsort(flat_r)
        return np.cumsum(flat_arr[idx]), idx, flat_r[idx]

    # cumulative sums
    θV_cumsum, idx, flat_r = radial_cumsum(theta * V, center)
    θ2V_cumsum, _, _       = radial_cumsum(theta**2 * V, center)
    σ2V_cumsum, _, _       = radial_cumsum(sigma**2 * V, center)
    V_cumsum, _, _         = radial_cumsum(V, center)

    # averages
    θ̄ = θV_cumsum / V_cumsum
    θ2̄ = θ2V_cumsum / V_cumsum
    σ2̄ = σ2V_cumsum / V_cumsum

    # Buchert Q
    Q_cumsum = (2/3) * (θ2̄ - θ̄**2) - 2 * σ2̄

    # mask outside radius
    if rmax is not None:
        Q_cumsum[flat_r > rmax] = np.nan

    # reshape back to 3D grid
    Q_grid = np.empty_like(rad)
    Q_grid.ravel()[idx] = Q_cumsum
    return Q_grid

mu, lu, tu = 1.989e45, 3.085678e19, 31557600e6
G = 6.6742e-11*((mu*tu**2)/(lu**3))
c = 299792458*(tu/lu)
H0 = (tu/lu)*H_0
t_0 = 2/(3*H0)
a = lambda t: (t/t_0)**(2/3)
a_dt = lambda t: 2/3*t**(-1/3)*t_0**(-2/3)
theta = lambda t: 3*a_dt(t)/a(t)
t = np.array([t_0*(1/(1+z))**(3/2) for z in (z_i, z_f)])
# --- setup common coords ---
X, Y, Z = np.meshgrid(coords, coords, coords)
rad = np.sqrt(X**2 + Y**2 + Z**2)
center = (g_size//2, g_size//2, g_size//2)

# --- compute Q for all datasets ---
Q_LTB_f    = compute_Q_grid(theta_f*theta(t[1]), sigma_f*theta(t[1])/3, V_f, center, rad)
Q_LTB_i    = compute_Q_grid(theta_i*theta(t[0]), sigma_i*theta(t[0])/3, V_i, center, rad)
Q_sim_f    = compute_Q_grid(runs["sim_f"][1]*theta(t[1]), runs["sim_f"][2]*theta(t[1])/3, runs["sim_f"][4], center, rad)
Q_sim_i    = compute_Q_grid(runs["sim_i"][1]*theta(t[0]), runs["sim_i"][2]*theta(t[0])/3, runs["sim_i"][4], center, rad)
Q_simic_f  = compute_Q_grid(runs["simic_f"][1]*theta(t[1]), runs["simic_f"][2]*theta(t[1])/3, runs["simic_f"][4], center, rad)
Q_simic_i  = compute_Q_grid(runs["simic_i"][1]*theta(t[0]), runs["simic_i"][2]*theta(t[0])/3, runs["simic_i"][4], center, rad)

# --- slot into dict like the others ---
quantities["Backreaction"] = {
    "LTB": (Q_LTB_f, Q_LTB_i),
    "Simsilun": (Q_sim_f, Q_sim_i),
    "Simsilun(IC)": (Q_simic_f, Q_simic_i)
}
labels["Backreaction"] = r"$\mathcal{Q}$"


# --- Plotting ---
for name, q in quantities.items():
    # Build grid with difference row centered (first column empty)
    grid = [
        [q["LTB"][1], q["Simsilun"][1],               q["Simsilun(IC)"][1]],
        [q["LTB"][0], q["Simsilun"][0],               q["Simsilun(IC)"][0]],
        [None,        q["LTB"][0] - q["Simsilun"][0], q["LTB"][0] - q["Simsilun(IC)"][0]]
    ]
    plot_grid(name, grid, coords, labels[name])
