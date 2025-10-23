import numpy as np

#data = np.load("data/jusilun_output/initial_vals_043.npz")
#np.savetxt("data/ics/rho_i_new", data["rho"][:-1]/data["rho"][-1])
#np.savetxt("data/ics/V_i_new", np.ones(data["rho"].shape[0]-1))

#data = np.load("../../Downloads/delta.npy")
#rho = data.reshape(256*256*256)
#np.savetxt("data/ics/rho_i_newnew", rho)
#np.savetxt("data/ics/V_i_newnew", np.ones(rho.shape[0]))

#jus = np.load("data/jusilun_output/final_vals_043.npz")
sil = np.loadtxt("data/simsilun_output/params")
#print(jus["rho"][-1])
#print(jus["rho"][:-1]/jus["rho"][-1])
print(np.max(sil[:, 0]))

import matplotlib.pyplot as plt
#plt.imshow((jus["rho"][:-1]/jus["rho"][-1]).reshape(256,256,256)[:,:,128])

#plt.colorbar()
#plt.show()

plt.imshow(sil[:, 0].reshape(256,256,256)[:,:,128])

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
#np.savez("data/simsilun_output/params_new", **alle)

# Volume-weighted means
def gm(X, V):
    return np.sum(X*V)/np.sum(V)

# Density parameters
H = gm(theta, V) / 3
Omega_m = gm(rho, V) / (3 * H ** 2)
Omega_Q = (gm(sigma ** 2, V) + 1 / 9 * gm(theta ** 2, V) - H ** 2) / H ** 2
R = 2*gm(rho, V) + 6*gm(sigma**2, V) - 2/3*gm(theta**2, V) + 2*0
Omega_K = -R / (6 * H ** 2)


mu = 1.989e45  # 10^15 solar masses
lu = 3.085678e19  # 10 kpc
tu = 31557600.0 * 1e6  # 1 mega years
c = 299792458.0 * tu / lu



print("Omega_m_f =", Omega_m)
print("Omega_Q_f =", Omega_Q)
print("Omega_K_f =", Omega_K)
print("Omega_T_f =", Omega_m + Omega_Q + Omega_K)
print("H", H*lu/tu*c)
