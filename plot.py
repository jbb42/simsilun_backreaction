import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if available
import matplotlib.pyplot as plt
import numpy as np

def plot_grid(g_size):
    initial = np.load("./data/jusilun_output/initial_vals_009.npz")
    final = np.load("./data/jusilun_output/final_vals_009.npz")
    rho_i = initial["rho"]
    rho_i_bg = rho_i[-1]
    rho_i = rho_i[:-1].reshape(g_size,g_size,g_size)
    rho_f = final["rho"]
    rho_f_bg = rho_f[-1]
    rho_f = rho_f[:-1].reshape(g_size,g_size,g_size)

    plt.figure()
    plt.imshow(rho_i[:,:,g_size//2]/rho_i_bg)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(rho_f[:,:,g_size//2]/rho_f_bg)
    plt.colorbar()
    plt.show()
    return
plot_grid(64)

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

def plot_grid_3d(g_size):
    # Load density data
    final = np.load("./data/jusilun_output/final_vals_009.npz")
    rho_f = final["rho"]
    rho_f_bg = rho_f[-1]
    rho_f = rho_f[:-1].reshape(g_size, g_size, g_size) / rho_f_bg

    # Coordinates
    x, y, z = np.mgrid[0:g_size, 0:g_size, 0:g_size]

    # Percentile cutoffs to focus on interesting range
    vmin = np.percentile(rho_f, 60)
    vmax = np.percentile(rho_f, 99)

    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=rho_f.flatten(),
        isomin=vmin,
        isomax=vmax,
        opacityscale=[  # alpha mapping
            [0.0, 0.0],   # lowest values fully transparent
            [0.3, 0.1],
            [0.6, 0.4],
            [0.8, 0.7],
            [1.0, 1.0],   # highest values fully opaque
        ],
        surface_count=20,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    fig.update_layout(
        title="Final Density Field (Volume Rendering with Transparency)",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='cube'
        ),
    )

    fig.show()

plot_grid_3d(64)
