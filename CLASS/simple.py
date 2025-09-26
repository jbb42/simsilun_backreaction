import numpy as np
from matplotlib import pyplot as plt
dat = np.load('/home/jbb/Downloads/S-GenIC/ICs/density64.npz')
rho   = dat['rho']      # (64,64,64)
delta = dat['delta']    # (64,64,64)
print(rho.shape, delta.mean(), dat['BoxSize'])

plt.imshow(delta[32,:,:])
plt.colorbar()
plt.show()