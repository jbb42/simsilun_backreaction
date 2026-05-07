#import Pkg; Pkg.add.(["DifferentialEquations", "Plots", "LaTeXStrings", "PGFPlotsX", "Interpolations", "QuadGK", "ForwardDiff", "NPZ", "Healpix"])
using DifferentialEquations
using Plots
using LaTeXStrings
using PGFPlotsX
using Interpolations
using QuadGK
using ForwardDiff

# Plot as .tex (Tikz) files
#pgfplotsx()
#push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepackage{amsmath}")
gr()
default(linewidth=1, framestyle=:box, grid=true, label=nothing, legend=:topright)

#=============================================================================#
# Defining basic constants and cosmological parameters
#=============================================================================#

# Physical constants and parameters
# Fundamental constants
const H_0 = 71.58781594e-3   # Hubble constant [1/Gyr]
const c = 306.5926758        # Speed of light [Mpc/Gyr]
const G_N = 4.498234911e-15  # G in Mpc^3/(M_sun*Gyr^2)

# Cosmological parameters
const Ω_Λ = 0.7
const Ω_m = 0.3

# LTB parameters
const r_b = 40.0
const k_max = 5.4e-8#5.4e-8
const n = 4
const m = 4

# Initial conditions
const a_i = 1/1200
const H_i = H_0 * sqrt(Ω_m * a_i^(-3) + Ω_Λ)
const Lambda = 3 * Ω_Λ * H_0^2
const rho_bg = 3 * Ω_m * H_0^2 / (8 * pi * G_N) / a_i^3

# Numerical solution of LTB dynamics
const r_grid = range(1e-6, r_b*1.25, length=1_000)
const N = length(r_grid)


function ltb(z=0)

    #=============================================================================#
    # Setting up LTB model
    #=============================================================================#

    # LTB functions
    K(r) = ifelse(r > r_b, 0.0, -r^2 * k_max * ((r/r_b)^n - 1)^m)
    K_r(r) = ifelse(r > r_b, 0.0, -2*r*k_max*((r/r_b)^n - 1)^m - r*k_max*n*m*((r/r_b)^n-1)^(m-1)*(r/r_b)^n)
    K_rr(r) = ifelse(r > r_b, 0.0, -2*k_max*((r/r_b)^n - 1)^m - k_max*n*m*(3+n)*((r/r_b)^n - 1)^(m-1)*(r/r_b)^n - k_max*n^2*m*(m-1)*((r/r_b)^n - 1)^(m-2)*(r/r_b)^(2n))

    M(r) = 4/3 * pi * G_N * r^3 * a_i^3 * rho_bg / c^2 * (1 + 3/5 * K(r) * c^2 / (a_i*H_i*r)^2)
    M_r(r) = 4/3 * pi * G_N * a_i^3 * rho_bg / c^2 * (3*r^2 + 3/5 * c^2/(a_i*H_i)^2 * (K(r) + r*K_r(r)))
    M_rr(r) = 4/3 * pi * G_N * a_i^3 * rho_bg / c^2 * (6*r + 3/5 * c^2/(a_i*H_i)^2 * (2*K_r(r) + r*K_rr(r)))

    # LCDM background
    t_of_a(a) = (2/3) * (1/H_0) / sqrt(Ω_Λ) * asinh(sqrt(Ω_Λ/Ω_m) * a^(3/2))
    a(t) = (Ω_m/Ω_Λ)^(1/3) * cbrt(sinh((3/2) * sqrt(Ω_Λ) * H_0 * t))^2
    a_t(t) = H_0 * sqrt(Ω_m/a(t) + Ω_Λ * a(t)^2)
    H_FLRW(z) = H_0 * sqrt(Ω_m * (1+z)^3 + Ω_Λ)

    # Numerical solution of LTB dynamics
    t_0 = t_of_a(1/(1+z))
    t_i = t_of_a(a_i)
    tspan = (t_i, t_0)

    # Initial conditions
    A_i(r) = a_i * r
    A_r_i(r) = a_i
    A_rr_i(r) = 0.0
    u0 = [A_i.(r_grid); A_r_i.(r_grid); A_rr_i.(r_grid)]

    # Parameters for ODEs
    p = (
        -K.(r_grid) * c^2,               # p[1]
        2*M.(r_grid) * c^2,              # p[2]
        fill(Lambda/3, length(r_grid)),  # p[3]
        2*M_r.(r_grid) * c^2,            # p[4]
        2*M.(r_grid) * c^2,              # p[5]
        -K_r.(r_grid) * c^2,             # p[6]
        fill(2*Lambda/3, length(r_grid)),# p[7]
        -K_rr.(r_grid) * c^2,            # p[8]
        2*M_rr.(r_grid) * c^2,           # p[9]
        -4*M_r.(r_grid) * c^2,           # p[10]
        4*M.(r_grid) * c^2               # p[11]
    )

    # Defining the system of ODEs for A, A_r, and A_rr
    function LTB_eq!(du, u, p, t)
        N = length(u) ÷ 3
        A   = @view u[1:N]
        Ar  = @view u[N+1:2N]
        Arr = @view u[2N+1:end]
        dA   = @view du[1:N]
        dAr  = @view du[N+1:2N]
        dArr = @view du[2N+1:end]

        @. dA = sqrt(p[1] + p[2]/A + p[3]*A^2)
        @. dAr = (p[4]/A - (p[5]*Ar)/(A^2) + p[6] + p[7]*A*Ar) / (2 * dA)
        @. dArr = ((p[8] + p[9]/A + (p[10]*Ar)/(A^2) + (p[11]*Ar^2)/(A^3) - (p[5]*Arr)/(A^2) + p[7]*Ar^2 + p[7]*A*Arr) - 2*dAr^2) / (2 * dA)
    end

        
    prob_LTB = ODEProblem(LTB_eq!, u0, tspan, p)
    
    # 1. Solve and extract the final state
    sol_LTB  = solve(prob_LTB, Tsit5(), reltol=1e-12, abstol=1e-12, save_everystep=false)
    t_final  = sol_LTB.t[end]
    u_final  = sol_LTB[end]

    du_final = similar(u_final)
    prob_LTB.f(du_final, u_final, prob_LTB.p, t_final)

    # 2. Generate Discrete Spatial Arrays (Vectors)
    # Using comprehensions to map the boundary condition across the grid
    A    = [r > r_b ? a(t_final) * r   : u_final[i]       for (i, r) in enumerate(r_grid)]
    A_r  = [r > r_b ? a(t_final)       : u_final[N + i]   for (i, r) in enumerate(r_grid)]
    A_rr = [r > r_b ? 0.0              : u_final[2N + i]  for (i, r) in enumerate(r_grid)]
    A_t  = [r > r_b ? a_t(t_final) * r : du_final[i]      for (i, r) in enumerate(r_grid)]
    A_tr = [r > r_b ? a_t(t_final)     : du_final[N + i]  for (i, r) in enumerate(r_grid)]

    # 3. Calculate physics quantities as fully discrete arrays
    # The `@.` macro automatically vectorizes the operations element-by-element
    ρ  = @. (c^2 / (4 * pi * G_N)) * (M_r(r_grid) / (A^2 * A_r))
    θ  = @. A_tr / A_r + 2 * A_t / A
    σ² = @. (1/3) * (A_tr / A_r - A_t / A)^2

    # 4. Initial Density Profile Array
    ρ_init = @. (c^2 / (4 * pi * G_N)) * (M_r(r_grid) / ((a(t_i) * r_grid)^2 * a(t_i)))

    # 5. Background Normalizations (Scalars)
    ρ_bg_init  = (c^2 / (4 * pi * G_N)) * (M_r(2r_b) / ((a(t_i) * 2r_b)^2 * a(t_i)))
    ρ_bg_final = (c^2 / (4 * pi * G_N)) * (M_r(2r_b) / ((a(t_final) * 2r_b)^2 * a(t_final)))

    # 6. Plotting (Passing the arrays directly)
    plot(r_grid, ρ_init ./ ρ_bg_init, 
        label="Initial Density Profile", 
        xlabel="r [Mpc]", 
        ylabel=L"\rho(t_i, r)/\rho(t_i, 2r_b)")

    plot!(r_grid, ρ ./ ρ_bg_final, 
        label="Final Density Profile", 
        xlabel="r [Mpc]", 
        ylabel=L"\rho(t_0, r)/\rho(t_0, 2r_b)")


    W = @. c^2 * (M(r_grid) / A^3 - M_r(r_grid) / (3 * A^2 * A_r))
    return @. (8 * pi * G_N / c^2) * ρ,  θ / c,  sqrt(σ² / 3) / c,  W / c^2

end

ρi, Θi, Σi, Wi = ltb(90)
ρf, Θf, Σf, Wf = ltb(0)

