import numpy as np
import matplotlib.pyplot as plt

# Parameters
g_size = 64
rb = 1.0   # comoving radius
a = 2.0    # scale factor

# --- Exact spherical volume ---
def V_sph(r):
    return (4/3) * np.pi * (a*r)**3

# --- Cartesian grid setup ---
dx = 2.0*rb / g_size

coords = np.linspace(-rb, rb, g_size, endpoint=False) + dx/2
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
r = np.sqrt(X**2 + Y**2 + Z**2)

cell_vol = a**3 * dx**3
r_flat = r.ravel()

# sort by radius
sort_idx = np.argsort(r_flat)
r_sorted = r_flat[sort_idx]
V_cumsum_sorted = np.cumsum(np.ones_like(r_sorted) * cell_vol)

# map back to grid
V_cumsum_grid = np.empty_like(r_flat)
V_cumsum_grid[sort_idx] = V_cumsum_sorted
V_cumsum_grid = V_cumsum_grid.reshape(r.shape)



# --- Plots ---
mid = g_size // 2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].imshow(V_sph(r[:,:,mid]), origin="lower",
                     extent=[-rb, rb, -rb, rb])
axes[0].set_title("Exact spherical volume (slice)")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(V_cumsum_grid[:,:,mid], origin="lower",
                     extent=[-rb, rb, -rb, rb])
axes[1].set_title("Cartesian cumsum volume (slice)")
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()

# --- Totals ---
print("Spherical total V(rb):", V_sph(rb))
print("Cartesian total:", V_cumsum_sorted[-1])
print("Relative difference:", (V_cumsum_sorted[-1]-V_sph(rb))/V_sph(rb))