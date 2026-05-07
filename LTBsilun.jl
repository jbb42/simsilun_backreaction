using Plots
include("LTB.jl")

using NPZ
using FFTW
using Random
using DelimitedFiles
using Interpolations
using DifferentialEquations

#====================================================================#
# Defining constants and paths
#====================================================================#

# Constants in cosmological units
const G = 4.498234911e-15  # G in Mpc^3/(M_sun*Gyr^2)
const c = 306.5926758      # Speed of light [Mpc/Gyr]
const κ = 8*pi*G/c^4

#====================================================================#
# Define simsilun ODE system
#====================================================================#

# Solve simsilun on grid
function simsilun_ode!(du, u, (Λ, active), t)
    @inbounds for i in axes(u, 2)
        if active === nothing || active[i] # Only evolve non-collapsed cells
            ρ, Θ, Σ, W, V = u[1,i], u[2,i], u[3,i], u[4,i], u[5,i]
            du[1,i] = -ρ*Θ
            du[2,i] = -(Θ^2)/3 - ρ/2 - 6*Σ^2 + Λ
            du[3,i] = -(2*Θ*Σ)/3 + Σ^2 - W
            du[4,i] = -Θ*W - ρ*Σ/2 - 3*Σ*W
            du[5,i] = V*Θ
        else
            du[1,i] = du[2,i] = du[3,i] = du[4,i] = du[5,i] = 0.0
        end
    end
end

#====================================================================#
# Define end time and callback for collape and end of evolution
#====================================================================#
function find_t_end(u0_bg, H0, Λ)
    cb = ContinuousCallback( # Stop background evolution when H = H0
        (u, t, integ) -> u[2] - 3*H0/c,
        nothing, terminate!;
        rootfind=SciMLBase.RightRootFind, save_positions=(false, true))

    prob_bg = ODEProblem(simsilun_ode!, u0_bg, (0.0, 1e4), (Λ, nothing))
    sol_bg = solve(prob_bg, Tsit5();
        callback=cb, reltol=1e-12, abstol=1e-14, save_everystep=false, dense=false, verbose=false)
    return sol_bg.t[end], sol_bg.u[end]
end

function collapse_condition(u, t, integrator)
    active = integrator.p.active # Get active cells
    @inbounds for i in axes(u, 2)
        if active[i] && u[2, i] <= 0.0 # If active cell has collapsed
            return true
        end
    end
    return false
    
end

function collapse_affect!(integrator)
    active = integrator.p.active
    u = integrator.u
    modified = false # Track if cells were modified

    @inbounds for i in axes(u,2)
        if active[i] && u[2,i] <= 0 # If active cell has collapsed
            active[i] = false
            u[2,i] = 0.0 # Set expansion rate explicitly to zero
            modified = true # Mark cell as modified
        end
    end

    modified && u_modified!(integrator, true)
end

cb_collapse = DiscreteCallback(collapse_condition, collapse_affect!;
                                     save_positions=(false, false))



#====================================================================#
# Solve LTB
#====================================================================#

const Ω_Λ = 0.7
const Ω_m = 0.3
const h = 0.7
const H_0 = h * 1.0227e-1 # Hubble constant in 1/Gyr

ρi, Θi, Σi, Wi = ltb(90)
ρf, Θf, Σf, Wf = ltb(0)

#====================================================================#
# Solve ODEs
#====================================================================#

function jusilun(ρi, Θi, Σi, Wi, Ω_m, Ω_Λ, h, zi=90.0)
    H0 = h * 1.0227e-1 # Hubble constant in 1/Gyr
    Λ = 3Ω_Λ * H0^2 / c^2 # Cosmological constant in 1/Mpc^2

    # Initial conditions for background and grid
    ρ_i = 3Ω_m * H0^2 / c^2 * (1 + zi)^3 # Initial background density

    Θ_i = 3 * H0 / c * sqrt(Ω_m * (1 + zi)^3 + Ω_Λ) # Initial background expansion rate
    u0_bg = [ρ_i, Θ_i, 0.0, 0.0, 1.0]

    # Pre-allocate a 5 × N_cells matrix for initial conditions
    u0 = Matrix{Float64}(undef, 5, length(ρi))

    δ = ρi ./ ρ_i .- 1.0 # Density contrast

    u0[1, :] = ρi
    u0[2, :] = Θi
    u0[3, :] = Σi
    u0[4, :] = Wi
    u0[5, :] = ones(length(ρi)) # Volume element V (from mass conservation)


    t_end, (ρ_f, Θ_f, _, _, _) = find_t_end(u0_bg, H0, Λ)

    p = (Λ=Λ, active=fill(true, size(δ)...))
    prob = ODEProblem(simsilun_ode!, u0, (0.0, t_end), p)
    sol = solve(prob, Tsit5(); callback=cb_collapse, reltol=1e-8, abstol=1e-10,
        save_everystep=false, dense=false, verbose=false)

    u_final = sol.u[end]
    
    return u_final, u0, ρ_i, ρ_f, Θ_i, Θ_f

end

function jusimsilun(ρi, Ω_m, Ω_Λ, h, zi=90.0)
    H0 = h * 1.0227e-1 # Hubble constant in 1/Gyr
    Λ = 3Ω_Λ * H0^2 / c^2 # Cosmological constant in 1/Mpc^2

    # Initial conditions for background and grid
    ρ_i = 3Ω_m * H0^2 / c^2 * (1 + zi)^3 # Initial background density
    Θ_i = 3 * H0 / c * sqrt(Ω_m * (1 + zi)^3 + Ω_Λ) # Initial background expansion rate
    u0_bg = [ρ_i, Θ_i, 0.0, 0.0, 1.0]

    δ = ρi ./ ρ_i .- 1.0 # Density contrast
    # Pre-allocate a 5 × N_cells matrix for initial conditions
    u0 = Matrix{Float64}(undef, 5, length(δ))
    
    # Fill the initial conditions
    @inbounds for i in eachindex(δ)
        δ_val = δ[i]
        u0[1, i] = ρ_i * (1 + δ_val)           # Density ρ
        u0[2, i] = Θ_i * (1 - δ_val / 3)       # Expansion rate Θ
        u0[3, i] = Θ_i * δ_val / 9             # Shear Σ
        u0[4, i] = -ρ_i * δ_val / 6            # Weyl curvature W
        u0[5, i] = 1.0 / (1 + δ_val)           # Volume element V (from mass conservation)
    end
    println("delta min and max: ", minimum(δ), " ", maximum(δ))
    t_end, (ρ_f, Θ_f, _, _, _) = find_t_end(u0_bg, H0, Λ)

    p = (Λ=Λ, active=fill(true, size(δ)...))
    prob = ODEProblem(simsilun_ode!, u0, (0.0, t_end), p)
    sol = solve(prob, Tsit5(); callback=cb_collapse, reltol=1e-8, abstol=1e-10,
        save_everystep=false, dense=false, verbose=false)

    u_final = sol.u[end]

    
    return u_final, u0, ρ_i, ρ_f, Θ_i, Θ_f

end


full_u_final, full_u0, full_ρ_i, full_ρ_f, full_Θ_i, full_Θ_f = jusilun(ρi, Θi, Σi, Wi, Ω_m, Ω_Λ, h, 90.0);

simple_u_final, simple_u0, simple_ρ_i, simple_ρ_f, simple_Θ_i, simple_Θ_f = jusimsilun(ρi, Ω_m, Ω_Λ, h, 90.0);

# ---------------------------------------------------------
# Plot 1: Initial Density Profile (z = 90)
# ---------------------------------------------------------
#=
plot(r_grid, ρi/ρi[end], 
    label="Exact initial",
    xlabel="r [Mpc]", ylabel=L"\rho / \rho_{bg}", 
    title="Density")

plot!(r_grid, simple_u0[1, :]/simple_ρ_i, 
    label="jusimsilun Initial")

plot!(r_grid, ρf/ρf[end], 
    label="Exact final")

plot!(r_grid, full_u_final[1, :]/full_ρ_f, 
    label="jusilun Final")

plot!(r_grid, simple_u_final[1, :]/simple_ρ_f, 
    label="jusimsilun Final")




plot(r_grid, Θi/Θi[end], 
    label="Exact initial",
    xlabel="r [Mpc]", ylabel=L"\Theta / \Theta_{bg}", 
    title="Expansion Rate")

plot!(r_grid, simple_u0[2, :]/simple_Θ_i, 
    label="jusimsilun Initial")

plot!(r_grid, Θf/Θf[end], 
    label="Exact final")

plot!(r_grid, full_u_final[2, :]/full_Θ_f, 
    label="jusilun Final")

plot!(r_grid, simple_u_final[2, :]/simple_Θ_f, 
    label="jusimsilun Final")




plot(r_grid, Σi, 
    label="Exact initial",
    xlabel="r [Mpc]", ylabel=L"\Sigma / \Sigma_{bg}", 
    title="Shear")

plot!(r_grid, simple_u0[3, :], 
    label="jusimsilun Initial")

plot!(r_grid, full_u0[3, :], 
    label="jusilun Initial")


plot!(r_grid, Σf, 
    label="Exact final")

plot!(r_grid, full_u_final[3, :], 
    label="jusilun Final")

plot!(r_grid, simple_u_final[3, :], 
    label="jusimsilun Final")
=#

plot(r_grid, Wi, 
    label="Exact initial",
    xlabel="r [Mpc]", ylabel=L"\W / \W_{bg}", 
    title="Weyl curvature")

plot!(r_grid, simple_u0[4, :], 
    label="jusimsilun Initial")

plot!(r_grid, full_u0[4, :], 
    label="jusilun Initial")


plot!(r_grid, Wf, 
    label="Exact final")

plot!(r_grid, full_u_final[4, :], 
    label="jusilun Final")

plot!(r_grid, simple_u_final[4, :], 
    label="jusimsilun Final")