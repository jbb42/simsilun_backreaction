import Pkg
using NPZ
using FFTW
using Random
using DelimitedFiles
using Interpolations
using LinearAlgebra
using Printf
using Base.Threads
using Plots
using DifferentialEquations

#====================================================================#
# Defining constants and paths
#====================================================================#

# Force CLASS to only use 1 thread to prevent oversubscription
ENV["OMP_NUM_THREADS"] = "1"

# Constants in cosmological units
const G = 4.498234911e-15  # G in Mpc^3/(M_sun*Gyr^2)
const c = 306.5926758      # Speed of light [Mpc/Gyr]
const κ = 8*pi*G/c^4

# Ensure output directories exist
mkpath("./data/jusilun_output")
mkpath("./initial_conditions")

#====================================================================#
# Generate initial conditions using CLASS
#====================================================================#

function run_class(Ωm, ΩΛ, Ωk, h, id; N=64, Lbox=256.0, zi=90.0)
    # Set wavenumber based on Nyquist frequency
    kNy = π * N / Lbox
    kmax = 2.0 * kNy

    # Write CLASS input file
    class_ini = "./initial_conditions/class_$(id).ini"
    class_out = "./initial_conditions/class_out_$(id)_"

    open(class_ini, "w") do io
        println(io, "output = mPk")
        println(io, "k_per_decade_for_pk = 32") # Smooth power spectrum
        println(io, "z_pk = $zi")
        println(io, "P_k_max_h/Mpc = $kmax")
        println(io, "Omega_m = $Ωm")
        println(io, "Omega_Lambda = $ΩΛ")
        println(io, "Omega_k = $Ωk")
        println(io, "h = $h")
        println(io, "n_s = 0.966")
        println(io, "sigma8 = 0.81")
        println(io, "root = $class_out")
    end

    # Run CLASS and read power spectrum
    run(`./class_public-3.3.4/class $class_ini`)
    ps = readdlm("$(class_out)00_pk.dat", comments=true, comment_char='#')

    # Clean up CLASS output files
    foreach(f -> rm(f, force=true),
            [class_ini, "$(class_out)parameters.ini", ["$(class_out)0$(i)_pk.dat" for i in 0:5]...])

    # Interpolated loglog power spectrum
    log_interp = linear_interpolation(log.(ps[:,1]), log.(ps[:,2]), extrapolation_bc=Line())

    # Return power spectrum function
    return k -> k > 0.0 ? exp(log_interp(log(k))) : 0.0
end

#====================================================================#
# Generate gaussian random field from power spectrum
#====================================================================#

function get_δ(Ωm, ΩΛ, Ωk, h, seed, id; N=64, Lbox=256.0, zi=90.0)
    P = run_class(Ωm, ΩΛ, Ωk, h, id; N=N, Lbox=Lbox, zi=zi)

    Random.seed!(seed)
    Vbox = Lbox^3
    kf   = 2π / Lbox
    
    # Use half grid along one axis to enforce Hermitian symmetry    
    Nx_half = N ÷ 2 + 1 
    δk = zeros(ComplexF64, Nx_half, N, N)

    # Box normalization and variance scaling
    amp_factor = (N^3) / sqrt(2.0 * Vbox) 

    @inbounds for ix in 0:Nx_half-1, iy in 0:N-1, iz in 0:N-1
        kx = kf * ix # Only positive kx values
        ky = kf * (iy <= N÷2 ? iy : iy-N)
        kz = kf * (iz <= N÷2 ? iz : iz-N)
        
        k = sqrt(kx^2 + ky^2 + kz^2)
        
        if k > 0.0
            δk[ix+1, iy+1, iz+1] = amp_factor * sqrt(P(k)) * (randn() + im*randn())
        end
    end

    # Inverse FFT to get real-space density field with Hermitian symmetry
    return irfft(δk, N) 
end

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
# Calculate Buchert averaged density parameters
#====================================================================#
function buchert(u, Λ)
    # Initialise accumulated sums
    sum_ρ = sum_θ = sum_θ2 = sum_σ2 = sum_V = 0.0

    # Add volume-weigthed contributions from each cell
    @inbounds @simd for i in axes(u, 2)
        ρ = u[1, i]
        θ = u[2, i]
        σ = u[3, i]
        V = u[5, i]
        
        sum_ρ += ρ * V
        sum_θ += θ * V
        sum_θ2 += θ^2 * V
        sum_σ2 += σ^2 * V
        sum_V += V
    end

    # Volume-weighted averages
    ρ_avg = sum_ρ / sum_V
    θ_avg = sum_θ / sum_V
    θ2_avg = sum_θ2 / sum_V
    σ2_avg = sum_σ2 / sum_V

    # Density parameters
    H = θ_avg / 3.0
    Ωm = ρ_avg / (3.0 * H^2)
    ΩQ = (σ2_avg - (1.0 / 9.0) * θ2_avg + H^2) / H^2
    ΩK = (-(1.0 / 3.0) * ρ_avg - σ2_avg + (1.0 / 9.0) * θ2_avg - Λ / 3.0) / H^2
    ΩΛ = Λ / (3.0 * H^2)
    ΩT = Ωm + ΩQ + ΩK + ΩΛ

    # Return as a NamedTuple for easy access
    return (Ωm=Ωm, ΩΛ=ΩΛ, ΩQ=ΩQ, ΩK=ΩK, ΩT=ΩT, H=H)
end

#====================================================================#
# Solve ODEs
#====================================================================#

function jusilun(Ωm, ΩΛ, Ωk, h, seed, id; N=64, Lbox=256.0, zi=90.0)
    δ = get_δ(Ωm, ΩΛ, Ωk, h, seed, id; N=N, Lbox=Lbox, zi=zi)
    H0 = h * 1.0227e-1 # Hubble constant in 1/Gyr
    Λ = 3ΩΛ * H0^2 / c^2 # Cosmological constant in 1/Mpc^2

    # Initial conditions for background and grid
    ρ_bg = 3Ωm * H0^2 / c^2 * (1 + zi)^3 # Initial background density
    Θ_bg = 3 * H0 / c * sqrt(Ωm * (1 + zi)^3 + ΩΛ) # Initial background expansion rate
    u0_bg = [ρ_bg, Θ_bg, 0.0, 0.0, 1.0]

    # Pre-allocate a 5 × N_cells matrix for initial conditions
    u0 = Matrix{Float64}(undef, 5, length(δ))
    
    # Fill the initial conditions
    @inbounds for i in eachindex(δ)
        δ_val = δ[i]
        u0[1, i] = ρ_bg * (1 + δ_val)           # Density ρ
        u0[2, i] = Θ_bg * (1 - δ_val / 3)       # Expansion rate Θ
        u0[3, i] = Θ_bg * δ_val / 9             # Shear Σ
        u0[4, i] = -ρ_bg * δ_val / 6            # Weyl curvature W
        u0[5, i] = 1.0                          # Volume element V
    end

    t_end, (ρ_bg_f, Θ_bg_f, _, _, _) = find_t_end(u0_bg, H0, Λ)
    println("Background reaches H=H₀ at t = $(round(t_end, digits=4)) Gyr")

    p = (Λ=Λ, active=fill(true, size(δ)...))
    prob = ODEProblem(simsilun_ode!, u0, (0.0, t_end), p)
    sol = solve(prob, Tsit5(); callback=cb_collapse, reltol=1e-8, abstol=1e-10,
        save_everystep=false, dense=false, verbose=false)

    u_final = sol.u[end]

    # Calculate Buchert averaged density parameters at initial and final times
    Ωi = buchert(u0, Λ)
    Ωf = buchert(sol.u[end], Λ)


    # Reshape final state into 5 × N grid and return as tuple of arrays
    return eachslice(reshape(u_final, 5, size(δ)...), dims=1), (ρ_bg_f, Θ_bg_f), (Ωi, Ωf)
end

@time (ρ, Θ, Σ, W, V), (ρ_bg_f, Θ_bg_f), (Ωi, Ωf) = jusilun(0.3, 0.7, 0.0, 0.7, 42, "test"; N=64, Lbox=256.0, zi=90.0)

heatmap(ρ[:,:,32]/ρ_bg_f, title="Density at z=0", xlabel="X", ylabel="Y", colorbar_title="ρ [M_sun/Mpc^3]")