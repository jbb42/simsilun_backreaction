from matplotlib import pyplot as plt
import numpy as np

data = np.loadtxt("density")

ic = np.reshape(data[:,0], [64,64])
sil = np.reshape(data[:,1], [64,64])
eds = np.reshape(data[:,2], [64,64])


fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# Plot each slice
im0 = ax[0].imshow(ic[:, :])
ax[0].set_title("IC")

im1 = ax[1].imshow(sil[:, :])
ax[1].set_title("SIL")

im2 = ax[2].imshow(eds[:, :])
ax[2].set_title("EDS")

# Add colorbars
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

