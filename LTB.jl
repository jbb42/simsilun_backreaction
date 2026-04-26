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
const k_max = 3e-8#5.4e-8#5.4e-8
const n = 4
const m = 4

# Initial conditions
const a_i = 1/(1+90)#1/1200
const H_i = H_0 * sqrt(Ω_m * a_i^(-3) + Ω_Λ)
const Lambda = 3 * Ω_Λ * H_0^2
const rho_bg = 3 * Ω_m * H_0^2 / (8 * pi * G_N) / a_i^3

# Numerical solution of LTB dynamics
const r_grid = range(1e-6, r_b, length=1_000)
const N = length(r_grid)

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
t_0 = t_of_a(1.0)
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
sol_LTB = solve(prob_LTB, Tsit5(), reltol=1e-12, abstol=1e-12, dense=true)

# Interpolate solution for A, A_r, A_rr and their time derivatives
# GEMINI INSERT START
# 1. Local cubic curve math
@inline function local_cubic(y1, y2, y3, y4, w)
    return y2 + 0.5 * w * (y3 - y1 + w * (2.0*y1 - 5.0*y2 + 4.0*y3 - y4 + w * (3.0*(y2 - y3) + y4 - y1)))
end

# 2. Dynamic, extrapolation-safe evaluator
function fast_eval(sol, t, r, r_grid, N, offset; t_deriv=false)
    # Evaluate time interpolation EXACTLY ONCE per call
    u = t_deriv ? sol(t, Val{1}) : sol(t)
    
    dr = step(r_grid)
    fi = (r - first(r_grid)) / dr + 1
    i = clamp(floor(Int, fi), 1, N-1)
    w = fi - i 
    
    # Use @inbounds for extra speed since we safely clamped `i`
    @inbounds begin
        y2 = u[i + offset]
        y3 = u[i + 1 + offset]
        
        # Ghost points for perfectly smooth edges
        y1 = i > 1   ? u[i - 1 + offset] : 2.0*y2 - y3
        y4 = i < N-1 ? u[i + 2 + offset] : 2.0*y3 - y2
    end
    
    return local_cubic(y1, y2, y3, y4, w)
end

# 3. Direct definitions - Clean, compact, and highly readable
A(t, r)    = r > r_b ? a(t)*r   : fast_eval(sol_LTB, t, r, r_grid, N, 0)
A_r(t, r)  = r > r_b ? a(t)     : fast_eval(sol_LTB, t, r, r_grid, N, N)
A_rr(t, r) = r > r_b ? 0.0      : fast_eval(sol_LTB, t, r, r_grid, N, 2N)
A_t(t, r)  = r > r_b ? a_t(t)*r : fast_eval(sol_LTB, t, r, r_grid, N, 0,  t_deriv=true)
A_tr(t, r) = r > r_b ? a_t(t)   : fast_eval(sol_LTB, t, r, r_grid, N, N,  t_deriv=true)
# LTB metric
gtt() = -c^2
grr(t, r) = A_r(t, r)^2 / (1 - K(r))
gθθ(t, r) = A(t, r)^2
gϕϕ(t, r, θ) = A(t, r)^2 * sin(θ)^2

# Calculate rho, theta and sigma of LTB
ρ(t, r) = @. (c^2 / (4*pi*G_N)) * (M_r(r) / (A(t,r)^2 * A_r(t,r)))
θ(t, r) = @. (A_tr(t,r)/A_r(t,r) + 2*A_t(t,r)/A(t,r))
σ²(t, r) = @. (1/3) * (A_tr(t,r)/A_r(t,r) - A_t(t,r)/A(t,r))^2
R(t,r, kt) = (8*pi*G_N/c^4) * ρ(t,r) * kt^2 * c^4 # R = R_\mu\nu k^\mu k^\nu

#plot(r_grid, ρ(t_i, r_grid)/ρ(t_i, 2r_b), label="Initial Density Profile", xlabel="r [Mpc]", ylabel=L"\rho(t_i, r)/\rho(t_i, 0)")
#plot!(r_grid, ρ(t_0, r_grid)/ρ(t_0, 2r_b), label="Final Density Profile", xlabel="r [Mpc]", ylabel=L"\rho(t_0, r)/\rho(t_i, 0)")


