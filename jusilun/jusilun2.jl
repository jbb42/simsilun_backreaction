import Pkg
# Pkg.add(["NPZ", "PyCall"]) # Uncomment if needed
using NPZ

const mu = 1.989e45   
const lu = 3.085678e19    
const tu = 31557600.0 * 1e6   

const G   = 6.6742e-11 * mu * tu^2 / lu^3
const c   = 299792458.0 * tu / lu
const κ   = 8*pi*G/c^4

function f2(x)
    y = round(x, digits=2)
    s = string(y)
    if occursin(".", s)
        decs = split(s, ".")[2]
        if length(decs) == 1
            s *= "0"
        end
    else
        s *= ".00"
    end
    return s
end

# 1. Moved derivatives outside and forced inlining for speed
@inline function derivatives(ρ, Θ, Σ, W, V, Λ)
    dρ = -ρ*Θ
    dΘ = -(1/3)*Θ^2 - (1/2)*ρ - 6*Σ^2 + Λ
    dΣ = -(2/3)*Θ*Σ + Σ^2 - W
    dW = -Θ*W - (1/2)*ρ*Σ - 3*Σ*W
    dV = V*Θ
    return dρ, dΘ, dΣ, dW, dV
end

# 2. Moved RK4 outside, avoiding closures
function rk4_step!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec, active_indices, dt, Λ)
    @inbounds for idx in eachindex(active_indices)
        i = active_indices[idx]
        
        ρ, Θ, Σ, W, V = ρ_vec[i], Θ_vec[i], Σ_vec[i], W_vec[i], V_vec[i]

        k1 = derivatives(ρ, Θ, Σ, W, V, Λ)
        k2 = derivatives(ρ + 0.5*dt*k1[1], Θ + 0.5*dt*k1[2], Σ + 0.5*dt*k1[3], W + 0.5*dt*k1[4], V + 0.5*dt*k1[5], Λ)
        k3 = derivatives(ρ + 0.5*dt*k2[1], Θ + 0.5*dt*k2[2], Σ + 0.5*dt*k2[3], W + 0.5*dt*k2[4], V + 0.5*dt*k2[5], Λ)
        k4 = derivatives(ρ + dt*k3[1], Θ + dt*k3[2], Σ + dt*k3[3], W + dt*k3[4], V + dt*k3[5], Λ)

        ρ_vec[i] = ρ + dt*(k1[1] + 2k2[1] + 2k3[1] + k4[1])/6
        Θ_vec[i] = Θ + dt*(k1[2] + 2k2[2] + 2k3[2] + k4[2])/6
        Σ_vec[i] = Σ + dt*(k1[3] + 2k2[3] + 2k3[3] + k4[3])/6
        W_vec[i] = W + dt*(k1[4] + 2k2[4] + 2k3[4] + k4[4])/6   
        V_vec[i] = V + dt*(k1[5] + 2k2[5] + 2k3[5] + k4[5])/6

        # Cap the expansion rate
        if Θ_vec[i] < 0.0
            Θ_vec[i] = 0.0
        end
    end
end

# 3. Created an allocation-free function for H_avg
function get_H_avg_rel(Θ_vec, V_vec, H_0_c)
    sum_TV = 0.0
    sum_V = 0.0
    @inbounds @simd for i in eachindex(Θ_vec, V_vec)
        sum_TV += Θ_vec[i] * V_vec[i]
        sum_V += V_vec[i]
    end
    return (sum_TV / sum_V) / 3.0 / H_0_c
end

# 4. Created an allocation-free function for dt
function get_dt(Θ_vec, Σ_vec, active_indices)
    dt_min = Inf
    @inbounds for i in active_indices
        val = 1e-3 / abs(Θ_vec[i] + Σ_vec[i]/3)
        if val < dt_min
            dt_min = val
        end
    end
    return dt_min
end
# 5. Evolve function in global scope
function evolve!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec, Λ, H_0_c)
    # Start with all cells active
    N_cells = length(ρ_vec)
    active_indices = collect(1:N_cells)
    
    H_avg_rel = get_H_avg_rel(Θ_vec, V_vec, H_0_c)
    step = 0

    # Stop if H reaches target, OR if all cells have collapsed
    while H_avg_rel >= 1.0 && !isempty(active_indices)
        dt = get_dt(Θ_vec, Σ_vec, active_indices)
        
        rk4_step!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec, active_indices, dt, Λ)
        
        # In-place remove any cells that collapsed during this step
        filter!(i -> Θ_vec[i] > 0.0, active_indices)
        
        H_avg_rel = get_H_avg_rel(Θ_vec, V_vec, H_0_c)

        if step % 100 == 0
            println("Step $step:\tH_avg/H_0 = $(H_avg_rel),\tfrozen=$(N_cells-length(active_indices)),\tdt=$(dt)")
        end

        step += 1
    end
    
    println("Final H_avg=$(H_avg_rel)H_0")
    return ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec
end

function jusilun(g_size, H_0_km_s_Mpc, Ω_m, Ω_Λ, z_i)
    H_0 = tu / lu * H_0_km_s_Mpc
    Λ   = 3 * Ω_Λ * H_0^2 / c^2
    ρ_0 = 3 * Ω_m * H_0^2 / c^2

    δ_i = npzread("./data/ics/delta.npy")
    δ_i = reshape(δ_i, g_size^3)
    push!(δ_i, 0.0) 

    ρ_bg_i = ρ_0 * (1 + z_i)^3
    Θ_bg_i = 3 * (H_0/c) * sqrt(Ω_m * (1 + z_i)^3 + Ω_Λ + (1 - Ω_m - Ω_Λ) * (1 + z_i)^2)

    # Initial allocations are fine here because they only happen once
    ρ_vec = ρ_bg_i .* (1 .+ δ_i)
    Θ_vec = Θ_bg_i .* (1 .- δ_i ./ 3)
    Σ_vec = Θ_bg_i .* δ_i ./ 9
    W_vec = .-(ρ_bg_i .* δ_i) ./ 6
    V_vec = ones(length(ρ_vec))  

    basepath_out = "./data/jusilun_output/i_m$(f2(Ω_m))_L$(f2(Ω_Λ))"
    file_num = 0
    filename_out = basepath_out * "_n" * lpad(string(file_num), 2, '0') * ".npz"
    while isfile(filename_out)
        file_num += 1
        filename_out = basepath_out * "_n" * lpad(string(file_num), 2, '0') * ".npz"
    end
    
    npzwrite(filename_out, Dict(
        "rho" => ρ_vec, "theta" => Θ_vec, "sigma" => Σ_vec, "W" => W_vec, "V" => V_vec,
        "H_0" => H_0_km_s_Mpc, "Omega_m" => Ω_m, "Omega_Lambda" => Ω_Λ, "z_i" => z_i
    ))

    # Pass the pre-calculated H_0 / c constant to avoid recomputing it
    H_0_c = H_0 / c 
    ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec = evolve!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec, Λ, H_0_c)
    
    basepath_final = "./data/jusilun_output/f_m$(f2(Ω_m))_L$(f2(Ω_Λ))"
    filename_final = basepath_final * "_n" * lpad(string(file_num), 2, '0') * ".npz"

    npzwrite(filename_final, Dict(
        "rho" => ρ_vec, "theta" => Θ_vec, "sigma" => Σ_vec, "W" => W_vec, "V" => V_vec,
        "H_0" => H_0_km_s_Mpc, "Omega_m" => Ω_m, "Omega_Lambda" => Ω_Λ, "z_i" => z_i
    ))
end

@time jusilun(64, 69, 0.31, 0.69, 90)