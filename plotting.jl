using Plots
include("LTB.jl")
include("jusilun.jl")

# 1. Define your independent parameters
physical_size = 2.5*r_b  # Total physical width of the box
grid_size = 64       # Resolution (128x128x128 looks smooth and renders instantly)

# 2. Create the coordinate axis
# This creates an axis from -0.625 to 0.625
half_size = physical_size / 2.0
coords = range(-half_size, half_size, length=grid_size)

# 3. Generate the 3D cube of distances to the center
r_cube = [sqrt(x^2 + y^2 + z^2) for x in coords, y in coords, z in coords]

# 4. Extract the central 2D slice to look at it
mid_idx = grid_size ÷ 2
central_slice = ρ.(t_i, r_cube[:, :, mid_idx]) ./ ρ(t_i, 2r_b)
#central_slice = θ.(t_0, r_cube[:, :, mid_idx]) ./ θ(t_0, 2r_b)

function get_δ(Ωm, ΩΛ, Ωk, h, seed, id; N=N, Lbox=Lbox, zi=zi)
    return (ρ.(t_i, r_cube)/ρ(t_i, 2r_b) .-1)
end

(ρ_cube, θ_cube, σ_cube, W_cube, V_cube), (ρ_bg_f, Θ_bg_f), (Ωi, Ωf) = jusilun(0.3, 0.7, 0.0, 0.7, 12345, 1; N=grid_size, Lbox=256.0, zi=90, headless=false)

# 1. Plot the Distance geometric slice
p1 = heatmap(coords, coords, central_slice, 
    title="Density analytical",
    xlabel="x [Mpc]", ylabel="y [Mpc]",
    aspect_ratio=:equal, 
    xlims=(-half_size, half_size),
    ylims=(-half_size, half_size))
display(p1)

# 2. Extract and normalize the density slice
density_slice = ρ_cube[:, :, mid_idx] ./ ρ_bg_f

# 3. Plot the Density Contrast
p2 = heatmap(coords, coords, density_slice, 
    title="Density simsilun",
    xlabel="x [Mpc]", ylabel="y [Mpc]",
    aspect_ratio=:equal, 
    xlims=(-half_size, half_size),
    ylims=(-half_size, half_size))
display(p2)



#=
Egentlig vil det være helt fint bare at plotte 1d, for LTB er jo sfærisk symmetrisk.
Så er det også lettere at sammenligne resultaterne med simsilun, og måske finde en smart måde
at vise afvigelserne på.
=#

using Plots
using Statistics

function plot_1D_radial_comparison(r_cube, sim_cube, ltb_cube, quantity_name; sample_rate=10)
    # 1. Flatten the 3D arrays into 1D vectors
    r_flat = vec(r_cube)
    sim_flat = vec(sim_cube)
    ltb_flat = vec(ltb_cube)
    
    # 2. Calculate the Fractional Relative Difference
    # We add a tiny number (eps) to the denominator to prevent division by zero in empty regions
    rel_diff = @. (sim_flat - ltb_flat) / (ltb_flat + eps(Float64))
    
    # 3. Sort by radius so we can draw clean lines
    sort_idx = sortperm(r_flat)
    r_sorted = r_flat[sort_idx]
    sim_sorted = sim_flat[sort_idx]
    ltb_sorted = ltb_flat[sort_idx]
    rel_diff_sorted = rel_diff[sort_idx]
    
    # 4. Downsample for the scatter plots (plotting every point makes the file huge)
    # Taking every Nth point keeps the shape but saves rendering time
    r_samp = r_sorted[1:sample_rate:end]
    sim_samp = sim_sorted[1:sample_rate:end]
    rel_samp = rel_diff_sorted[1:sample_rate:end]

    # --- TOP PANEL: Absolute Physical Values ---
    p_top = scatter(r_samp, sim_samp, 
        label="Simsilun (Grid Cells)", 
        markersize=1.5, markeralpha=0.3, markerstrokewidth=0, color=:blue,
        ylabel=quantity_name,
        title="1D Radial Profile: $quantity_name",
        legend=:topright,
        framestyle=:box)
        
    # Overlay the exact analytical LTB solution as a sharp, solid red line
    plot!(p_top, r_sorted, ltb_sorted, 
        label="LTB (Exact)", 
        linewidth=2, color=:red)

    # --- BOTTOM PANEL: Relative Difference ---
    p_bottom = scatter(r_samp, rel_samp,
        label=nothing, 
        markersize=1.5, markeralpha=0.3, markerstrokewidth=0, color=:purple,
        xlabel="Radius r [Mpc]", 
        ylabel="Fractional Error\n(Sim - LTB) / LTB",
        framestyle=:box)
        
    # Add a baseline at 0.0 (perfect agreement)
    hline!(p_bottom, [0.0], color=:black, linewidth=1.5, linestyle=:dash, label=nothing)

    # --- COMBINE INTO A SINGLE FIGURE ---
    # layout = @layout [a{0.7h}; b{0.3h}] makes the top plot taller than the bottom one
    fig = plot(p_top, p_bottom, 
        layout=grid(2, 1, heights=[0.7, 0.3]), 
        size=(800, 600), link=:x, margin=5Plots.mm)
        
    return fig
end

# Example usage assuming you have your cubes calculated:
fig = plot_1D_radial_comparison(r_cube, ρ_cube/ρ_bg_f, ρ.(t_0, r_cube) ./ ρ(t_0, 2r_b), "Density Contrast")
display(fig)
