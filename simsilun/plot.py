from matplotlib import pyplot as plt
import numpy as np

data = np.loadtxt("density")

ic = np.reshape(data[:,0], [64,64])
sil = np.reshape(data[:,1], [64,64])
eds = np.reshape(data[:,2], [64,64])

fig, ax = plt.subplots(1,3,figsize=(10,4))
ax[0].imshow(ic, label="IC")
ax[1].imshow(sil, label="SIL")
ax[2].imshow(eds, label="EDS")
plt.show()
