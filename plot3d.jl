using GLMakie
using NPZ

function plot_initial_vs_final_independent(rho_i, rho_f)
    nx, ny, nz = size(rho_i)
    
    # 1. Increased global fontsize and reduced overall figure height to tighten layout
    fig = Figure(size = (1000, 600), figure_padding = 2, fontsize = 20)
    
    origin = Vec3f(0.0, 0.0, 0.0)
    widths = Vec3f(nx, ny, nz)
    cube_box = Rect3f(origin, widths)

    # --- Initial State ---
    ax_i = Axis3(fig[1, 1], 
                 title = L"$z=90$", 
                 titlesize = 24,     # 2. Made title explicitly larger
                 aspect = :data, elevation = pi/6, azimuth = pi/4)
    
    hidedecorations!(ax_i)
    hidespines!(ax_i)
                 
    vol_i = volume!(ax_i, rho_i, algorithm = :mip, colormap = :viridis)
    wireframe!(ax_i, cube_box, color = :black, linewidth = 1)
    
    # 3. 'height' controls the thickness of the horizontal color strip
    Colorbar(fig[2, 1], vol_i, vertical = false, height = 15)

    # --- Final State ---
    ax_f = Axis3(fig[1, 2], 
                 title = L"$z=0$", 
                 titlesize = 24, 
                 aspect = :data, elevation = pi/6, azimuth = pi/4)
                 
    hidedecorations!(ax_f)
    hidespines!(ax_f)
                 
    vol_f = volume!(ax_f, rho_f, algorithm = :mip, colormap = :viridis)
    wireframe!(ax_f, cube_box, color = :black, linewidth = 1)
    
    Colorbar(fig[2, 2], vol_f, vertical = false, height = 15)
    
    # 4. Strip out dead whitespace between rows and columns
    rowgap!(fig.layout, -20)   # Closes the vertical gap between the 3D plots and colorbars
    colgap!(fig.layout, 0)  # Brings the left and right columns closer together
    
    return fig
end

# Load data
data = npzread("./output_data/initialandfinal.npz")
rho_i = data["rho_i"] ./ data["rho_bg_i"]
rho_f = data["rho_f"] ./ data["rho_bg_f"]

# Generate the figure
fig = plot_initial_vs_final_independent(rho_i, rho_f)

save("initial_final.png", fig, px_per_unit = 4) # 4x resolution (roughly 300-600 DPI)