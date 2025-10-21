import numpy as np

data = np.load("data/jusilun_output/initial_vals_043.npz")
np.savetxt("data/ics/rho_i_new", data["rho"][:-1]/data["rho"][-1])
np.savetxt("data/ics/V_i_new", np.ones(data["rho"].shape[0]-1))

jus = np.load("data/jusilun_output/final_vals_043.npz")
sil = np.loadtxt("data/simsilun_output/params")
print(jus["rho"][-1])
print(jus["rho"][:-1]/jus["rho"][-1])
print(np.max(sil[:, 0]))

import matplotlib.pyplot as plt
plt.imshow((jus["rho"][:-1]/jus["rho"][-1]).reshape(128,128,128)[:,:,64])

plt.colorbar()
plt.show()

plt.imshow(sil[:, 0].reshape(128,128,128)[:,:,64])

plt.colorbar()
plt.show()



rho = sil[:,0]
theta = sil[:,1]
sigma = sil[:,2]
weyl = sil[:,3]
V = sil[:,4]

def gm(arr, V=V):
    return sum(arr*V) / sum(V)

Q = 2 / 3 * (gm(theta**2) - gm(theta)**2) - 2 * gm(sigma**2)
R = 2 * gm(rho) + 6 * gm(sigma**2) - 2 / 3 * gm(theta**2)
H = sum(theta*V) / sum(V) / 3



alle = dict(zip(["rho", "theta", "sigma", "weyl", "V", "Q", "R", "H"], [rho, theta, sigma, weyl, V, Q, R, H]))
np.savez("data/simsilun_output/params_new", **alle)
