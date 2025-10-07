using Statistics
using PyPlot
pygui(true)
using NPZ
using Base.Threads: @threads

# Parameters & units (feel free to change)
g_size = 64
H_0_km_s_Mpc = 70.0
Ω_m = 0.3
Ω_Λ = 0.2#1.0 - Ω_m
z_i = 90.0
z_f = 0.0

# Simsilun units
mu = 1.989e45   # 10^15 solar masses
lu = 3.085678e19    # 10 kpc
tu = 31557600.0 * 1e6   # 1 mega years

G   = 6.6742e-11 * mu * tu^2 / lu
c   = 299792458.0 * tu / lu

# Geometric units
H_0 = tu / lu * H_0_km_s_Mpc
Λ   = 3 * Ω_Λ * H_0^2 / c^2
ρ_0 = 3 * Ω_m * H_0^2 / c^2 

# Initial conditions
#δ_i = npzread("data/np_arrays_LTB/grid_rho0.npy").-1.0
δ_i = npzread("data/ics/delta.npy")
δ_i = reshape(δ_i, g_size^3)
push!(δ_i, 0.0) # add one more point for the background
#δ_i = [exp(-((x - g_size/2)^2 + (y - g_size/2)^2 + (z - g_size/2)^2) / 100.0) for z in 1:g_size, y in 1:g_size, x in 1:g_size]
#δ_i = vec(δ_i)  # same order as jusilun.jl
ρ_bg_i = ρ_0 * (1 + z_i)^3
Θ_bg_i = 3 * (H_0/c) * sqrt(Ω_m * (1 + z_i)^3 + Ω_Λ + (1 - Ω_m - Ω_Λ) * (1 + z_i)^2)

ρ_vec = ρ_bg_i .* (1 .+ δ_i)
Θ_vec = Θ_bg_i .* (1 .- δ_i ./ 3)
Σ_vec = Θ_bg_i .* δ_i ./ 9
W_vec = .-(ρ_bg_i .* δ_i) ./ 6
V_vec = ones(length(ρ_vec))  # CHANGE THIS

# Plot density
function plot_density(ρ_vec; grid=g_size, plt_title=" ")
    grid_img = reshape(ρ_vec[1:end-1], grid, grid, grid)
    figure(figsize=(6,6))
    imshow(grid_img[:,:,32]'/ρ_vec[end-1]; origin="lower", cmap="viridis", aspect="equal")
    colorbar()
    title(plt_title)
    tight_layout()
    show()
end

# Plot initial density
plot_density(ρ_vec, grid=g_size, plt_title="Initial Density Distribution at z=$z_i")

# RK4 step
function rk4_step!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec, dt, Λ, frozen)
    N = length(ρ_vec)

    # derivative function
    function derivatives(ρ, Θ, Σ, W, V)
        dρ = -ρ*Θ
        dΘ = -(1/3)*Θ^2 - (1/2)*ρ - 6*Σ^2 + Λ
        dΣ = -(2/3)*Θ*Σ + Σ^2 - W
        dW = -Θ*W - (1/2)*ρ*Σ - 3*Σ*W
        dV = V*Θ
        return dρ, dΘ, dΣ, dW, dV
    end

    @inbounds @threads for i in 1:N
        if frozen[i]; continue; end

        ρ, Θ, Σ, W, V = ρ_vec[i], Θ_vec[i], Σ_vec[i], W_vec[i], V_vec[i]

        # RK4 stages
        k1 = derivatives(ρ, Θ, Σ, W, V)
        k2 = derivatives(ρ + 0.5*dt*k1[1], Θ + 0.5*dt*k1[2],
                         Σ + 0.5*dt*k1[3], W + 0.5*dt*k1[4], V + 0.5*dt*k1[5])
        k3 = derivatives(ρ + 0.5*dt*k2[1], Θ + 0.5*dt*k2[2],
                         Σ + 0.5*dt*k2[3], W + 0.5*dt*k2[4], V + 0.5*dt*k2[5])
        k4 = derivatives(ρ + dt*k3[1], Θ + dt*k3[2],
                         Σ + dt*k3[3], W + dt*k3[4], V + dt*k3[5])

        # RK4 update
        ρ_vec[i] = ρ + dt*(k1[1] + 2k2[1] + 2k3[1] + k4[1])/6
        Θ_vec[i] = Θ + dt*(k1[2] + 2k2[2] + 2k3[2] + k4[2])/6
        Σ_vec[i] = Σ + dt*(k1[3] + 2k2[3] + 2k3[3] + k4[3])/6
        W_vec[i] = W + dt*(k1[4] + 2k2[4] + 2k3[4] + k4[4])/6   
        V_vec[i] = V + dt*(k1[5] + 2k2[5] + 2k3[5] + k4[5])/6

        # collapse → freeze
        if Θ_vec[i] < 0.0
            Θ_vec[i] = 0.0
            frozen[i] = true
        end
    end
end

# Evolve universe until <H> = H_0
function evolve!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec; Λ)
    frozen = falses(length(ρ_vec))
    H_avg = mean(Θ_vec) / 3
    step = 0
    while H_avg >= H_0/c
        dt = min(1e-3 / maximum(Θ_vec .+ Σ_vec ./ 3))
        if H_avg/(H_0/c) < 1.001
            dt *= 0.01
        end

        rk4_step!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec, dt, Λ, frozen)
        H_avg = mean(Θ_vec)/3 

        if step % 100 == 0
            println("Step $step:\tH_avg/H_0 = $(H_avg/H_0*c),\tfrozen=$(count(frozen))")
        end

        step += 1
        # all frozen? then done
        if all(frozen)
            break
        end
    end
    println("H_avg=$(mean(Θ_vec)/3/H_0*c)H_0")
    return ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec
end

ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec = evolve!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec; Λ=Λ)


plot_density(ρ_vec, grid=g_size, plt_title="Final Density Distribution at z=$z_f")