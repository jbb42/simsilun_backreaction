import numpy as np
import matplotlib.pyplot as plt

def radial_cumsum(arr_3d, center, r_max=32):
    """Cumulative sum of array elements sorted by radius from center."""
    g_size = arr_3d.shape[0]
    coords = (np.arange(g_size) - (g_size-1)/2)
    X, Y, Z = np.meshgrid(coords, coords, coords)
    rad = np.sqrt(X**2 + Y**2 + Z**2)
    arr = arr_3d.ravel()
    r = rad.ravel()
    idx = np.argsort(r)
    cs = np.cumsum(arr[idx]).astype(float)
    cs[r[idx] > r_max] = np.nan
    return cs, idx

def reshape(original_shape, values, indices):
    out = np.empty(original_shape, float).ravel()
    out[indices] = values
    return out.reshape(original_shape)

# --- Constants ---
H_0, z_i, z_f = 70, 1100.0, 0.0
mu, lu, tu = 1.989e45, 3.085678e19, 31557600e6
G = 6.6742e-11*((mu*tu**2)/(lu**3))
c = 299792458*(tu/lu)
H0 = (tu/lu)*H_0
t_0 = 2/(3*H0)
a = lambda t: (t/t_0)**(2/3)
a_dt = lambda t: 2/3*t**(-1/3)*t_0**(-2/3)
theta = lambda t: 3*a_dt(t)/a(t)
t = np.array([t_0*(1/(1+z))**(3/2) for z in (z_i, z_f)])

# --- Load data ---
def load_params(z, all_ic):
    if z == 0:
        ts = 1
    elif z==1100:
        ts = 0
    else: print("Invalid z value")

    p = np.load(f"data/params_z{z}_all_ic_{all_ic}.npy").reshape(64,64,64,-1)
    θ = p[...,1]*theta(t[ts])
    σ = p[...,2]*theta(t[ts])/3
    V = p[...,4]
    return θ, σ, V

θ_i, σ_i, V_i = load_params(1100, True)
θ_f, σ_f, V_f = load_params(0, True)

# --- Helper to compute Q ---
def compute_Q(θ, σ, V):
    θV, idx = radial_cumsum(θ*V, np.array(θ.shape)//2)
    Vol, _  = radial_cumsum(V, np.array(θ.shape)//2)
    θ2V, _  = radial_cumsum(θ**2*V, np.array(θ.shape)//2)
    σ2V, _  = radial_cumsum(σ**2*V, np.array(θ.shape)//2)

    θ̄ = θV/Vol
    σ2 = σ2V/Vol
    θ2 = θ2V/Vol
    Q = 2/3*(θ2-θ̄**2) - 2*σ2
    return reshape(θ.shape, Q, idx)

Q_f = compute_Q(θ_f, σ_f, V_f)
Q_i = compute_Q(θ_i, σ_i, V_i)

# --- Plots ---
for Q, label in [(Q_f,"Q_f"), (Q_i,"Q_i")]:
    plt.imshow(Q[:,:,32]); plt.colorbar(); plt.title(label); plt.show()
