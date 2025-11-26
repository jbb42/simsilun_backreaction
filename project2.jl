using DifferentialEquations
using Plots
using LaTeXStrings
using PGFPlotsX
# Switch backend
pgfplotsx()

# Your plot settings stay mostly the same, but we can remove the font setting
# because PGFPlotsX uses your system LaTeX font by default (Computer Modern).
default(
    linewidth=2, 
    framestyle=:box, 
    grid=false,
    label=nothing
)


#c = 1#299792458
#G_N = 1#6.67430e-11


# From LTB bachelor
H_0 = 71.58781594e-3   # Hubble constant [1/Gyr]
c = 306.5926758        # Speed of light [Mpc/Gyr]
G_N = 4.498234911e-15    # G in Mpc^3/(M_sun*Gyr^2)

Ω_Λ = 0.7
Ω_m = 0.3
r_b = 100.0
r = range(1, r_b*1.2, 150)
k_max = 2e-7#1
n = m = 4
a_i = 1/1200

# Change to LCDM
t(a) = (2/3)*(1/H_0)/sqrt(Ω_Λ)*asinh(sqrt(Ω_Λ/Ω_m)*a^(3/2))
a(t) = (Ω_m/Ω_Λ)^(1/3)*sinh((3/2)*sqrt(Ω_Λ)*H_0*t)^(2/3)

t_0 = t(1.0)
H_i = H_0 * sqrt(Ω_m * a_i^(-3) + Ω_Λ)
t_i = t(a_i)
Lambda = 3 * Ω_Λ * H_0^2
rho = 3 * Ω_m * H_0^2 / (8 * pi * G_N) / a_i^3

k(r)   = @. ifelse(r > r_b, 0.0, -r^2 * k_max * ((r/r_b)^n - 1)^m)
M(r)   = @. 4/3*pi*G_N*r^3*a_i^3*rho/c^2*(1+3/5*k(r)*c^2/(a_i*H_i*r)^2)
k_r(r) = @. ifelse(r > r_b, 0.0, -2*r*k_max*((r/r_b)^n - 1)^m -r*k_max*n*m*((r/r_b)^n-1)^(m-1)*(r/r_b)^n)
M_r(r) = @. 4/3*pi*G_N*a_i^3*rho/c^2*(3*r^2+3/5*c^2/(a_i*H_i)^2*(k(r)+r*k_r(r)))
# Initial conditions

tspan = (t_i, t_0)
t_steps = [0.002, 0.01, 0.05, 0.2, 0.5, 1.0]*t_0 
# Initial conditions
A_i = collect(a_i .* r)
Ar_i = fill(a_i, length(r))
u0 = [A_i; Ar_i]

# --- CORRECTED PARAMETERS ---
# 1. We only need indices 1 through 7.
# 2. We ensure every k and M term is multiplied by c^2.
# 3. We ensure Lambda is NOT divided by c^2.
p = (
    -k(r) .* c^2,      # p[1]: Term 1 for A
    2 * M(r) .* c^2,   # p[2]: Term 2 for A
    Lambda/3,          # p[3]: Term 3 for A
    2 * M_r(r) .* c^2, # p[4]: Term 1 for A_r (Numerator)
    2 * M(r) .* c^2,   # p[5]: Term 2 for A_r (Numerator factor)
    -k_r(r) .* c^2,    # p[6]: Term 3 for A_r (Numerator)
    2 * Lambda/3       # p[7]: Term 4 for A_r (Numerator factor)
)

# --- CORRECTED ODE FUNCTION ---
function ode!(du, u, p, t)
    N = length(u) ÷ 2
    
    # Views
    A   = @view u[1:N]
    Ar  = @view u[N+1:end]
    dA  = @view du[1:N]
    dAr = @view du[N+1:end]

    # 1. Solve for A_t (dA/dt)
    # Equation: A_t = sqrt( -kc^2 + 2Mc^2/A + (Lambda/3)A^2 )
    @. dA = sqrt(p[1] + p[2]/A + p[3]*A^2)
    
    # 2. Solve for A_tr (dA_r/dt)
    # The Equation: 2*A_t * A_tr = (Numerator)
    # Numerator: (2M'c^2)/A - (2Mc^2)*Ar/A^2 + (-k'c^2) + (2Lambda/3)*A*Ar
    
    # Note fixes: 
    # - Added (* Ar) to the p[5] term
    # - Divides by (2 * dA) instead of reconstructing the square root manually
    @. dAr = (p[4]/A - (p[5]*Ar)/(A^2) + p[6] + p[7]*A*Ar) / (2 * dA)
end

# Solve...
prob = ODEProblem(ode!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=t_steps, reltol=1e-6, abstol=1e-6)



# 1. Initialize the plot
A_plot = Plots.plot(
    xlabel=L"r/r_b", 
    ylabel=L"A(t,r)/ar", 
    title=L"Areal radius $A(t,r)$",
    size = (400, 300),
    legend = :topright,
    legend_font_halign = :left
)
#plot!(xscale=:identity, yscale=:log10, minorgrid=true)

for t in t_steps
    # Get state vector at time t
    current_u = sol(t)
    N = length(current_u) ÷ 2
    current_A = @view current_u[1:N]
    Plots.plot!(
        r/r_b, 
        current_A./(a(t)*r), 
        label=L"t/t_0=%$(round(t/t_0, digits=3))",
        linewidth=1.0
    )
end

# 4. Display
display(A_plot)
savefig("radial_profile.tex")
savefig("radial_profile.pdf")




# 1. Initialize the plot
Ar_plot = Plots.plot(
    xlabel=L"r/r_b", 
    ylabel=L"A'(t,r)/a", 
    title=L"Derivative of areal radius $A'(t,r)$",
    size = (400, 300),
    legend = :bottomleft,
    legend_font_halign = :left
)
#plot!(xscale=:identity, yscale=:log10, minorgrid=true)

for t in t_steps
    # Get state vector at time t
    current_u = sol(t)
    N = length(current_u) ÷ 2
    current_Ar = @view current_u[N+1:end]
    Plots.plot!(
        r/r_b, 
        current_Ar./a(t), 
        label=L"t/t_0=%$(round(t/t_0, digits=3))",
        linewidth=1.0
    )
end

# 4. Display
display(Ar_plot)
savefig("derived_radial_profile.tex")
savefig("derived_radial_profile.pdf")

