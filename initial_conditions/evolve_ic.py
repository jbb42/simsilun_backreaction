import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import ones_like
from pygadgetreader import readsnap, readheader

snapshot = ('/home/jbb/Documents/simsilun_backreaction/initial_conditions/S-GenIC/ICs/ICs')

# Read positions for each particle type
pos_dm = readsnap(snapshot, 'pos', 'dm')
pos_gas = readsnap(snapshot, 'pos', 'gas')

# Read masses from header
massarr = readheader(snapshot, 'mass')
npart = readheader(snapshot, 'npart')
mass_dm = np.full(npart[1], massarr[1])
mass_gas = np.full(npart[0], massarr[0])

# Combine all positions and masses
positions = np.concatenate([pos_dm, pos_gas], axis=0)
masses = np.concatenate([mass_dm, mass_gas], axis=0)

# Grid setup
grid_size = 64
box_size = readheader(snapshot, 'boxsize')
cell_size = box_size / grid_size

# Normalize positions to grid coordinates
u = (positions / cell_size) % grid_size
i = np.floor(u).astype(np.int64)

# TSC kernel
def W(d):
    w = np.zeros_like(d)
    m1 = d < 0.5
    m2 = (d >= 0.5) & (d < 1.5)
    w[m1] = 0.75 - d[m1]**2
    w[m2] = 0.5 * (1.5 - d[m2])**2
    return w

# Initialize density grid
rho = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)

# Apply TSC mass assignment
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            cx = (i[:, 0] + dx + 0.5)
            cy = (i[:, 1] + dy + 0.5)
            cz = (i[:, 2] + dz + 0.5)
            wx = W(np.abs(u[:, 0] - cx))
            wy = W(np.abs(u[:, 1] - cy))
            wz = W(np.abs(u[:, 2] - cz))
            w = wx * wy * wz
            ix = (i[:, 0] + dx) % grid_size
            iy = (i[:, 1] + dy) % grid_size
            iz = (i[:, 2] + dz) % grid_size
            np.add.at(rho, (ix, iy, iz), masses * w)

# Normalize by cell volume
cell_volume = cell_size ** 3
rho /= cell_volume

# Compute mean density and density contrast
rho_bar = masses.sum() / (box_size ** 3)
delta = rho / rho_bar - 1.0

# Visualize central slice of density contrast
slice_index = grid_size // 2
plt.figure(figsize=(8, 6))
plt.imshow(delta[:, :, slice_index])
plt.colorbar(label='Density Contrast (δ)')
plt.title(f'Density Contrast Slice at Z={slice_index}')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.savefig('../plots/density_contrast_slice.png')
plt.show()

rho_i = delta + np.ones_like(64**3)
np.savetxt("../data/ics/rho_i", rho_i.reshape(grid_size ** 3))
V_i = np.ones_like(rho_i)
np.savetxt("../data/ics/V_i", V_i.reshape(grid_size ** 3))


import subprocess
def run_simsilun(z_start, z_end, H_0, use_all_ic=True):
    subprocess.run([
        "../simsilun/simsilun", str(z_start), str(z_end), str(H_0),
        ".true." if use_all_ic else ".false."
    ], check=True)
    return

z_i = readheader(snapshot, 'redshift')
z_f = 0
H0 = readheader(snapshot, 'hubble')*100

run_simsilun(z_i, z_f, H0, False)

p = np.loadtxt("../data/simsilun_output/params").reshape(-1, 5)
dens = p[:,0].reshape(grid_size,grid_size,grid_size)
plt.imshow(dens[:,:,slice_index+1])
plt.colorbar()
plt.show()

import plotly.graph_objects as go

x, y, z = np.indices(dens.shape)
fig = go.Figure(data=go.Volume(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=dens.flatten(),
    isomin=np.min(dens), isomax=np.max(dens),
    opacity=0.1,       # global transparency
    surface_count=20,  # how many contour surfaces
    colorscale="Viridis"
))
fig.show()
