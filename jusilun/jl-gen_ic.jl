using DelimitedFiles
using FFTW
using Random
using Interpolations
using NPZ

# Force CLASS to only use 1 thread
ENV["OMP_NUM_THREADS"] = "1"
const class_exec = "./class_public-3.3.4/class"

# Constants in cosmological units
const G = 4.498234911e-15  # G in Mpc^3/(M_sun*Gyr^2)
const c = 306.5926758      # Speed of light [Mpc/Gyr]
const κ = 8*pi*G/c^4

# =====================================================================
# 1. IC GENERATOR
# =====================================================================
function get_density_field(Om, Ob, OL, h, seed, run_id_str; N=64, Lbox=256.0, zstart=90.0)
    kNy = π * N / Lbox
    kmax = 2.0 * kNy
    
    ini_file = "class_$(run_id_str).ini"
    root_out = "class_out_$(run_id_str)_"
    
    open(ini_file, "w") do io
        println(io, "output = mPk")
        println(io, "z_pk = $zstart")
        println(io, "P_k_max_h/Mpc = $kmax") 
        println(io, "Omega_cdm = $(Om - Ob)")
        println(io, "Omega_b = $Ob")
        println(io, "Omega_Lambda = $OL")
        println(io, "h = $h")
        println(io, "n_s = 0.965")
        println(io, "sigma8 = 0.9")
        println(io, "root = $root_out")
    end

    run(`$class_exec $ini_file`)
    
    pk_file = "$(root_out)00_pk.dat"
    data = readdlm(pk_file, comments=true, comment_char='#')
    k_tab = data[:, 1]
    P_tab = data[:, 2]
    
    rm(ini_file, force=true)
    rm("$(root_out)parameters.ini", force=true)
    for i in 0:5
        rm("$(root_out)0$(i)_pk.dat", force=true)
    end

    Random.seed!(seed)
    Vbox = Lbox^3
    kf   = 2π / Lbox
    P_interp = LinearInterpolation(k_tab, P_tab, extrapolation_bc=0.0)
    δk = zeros(ComplexF64, N, N, N)
    amp_factor = (N^3) / sqrt(2.0 * Vbox)

    @inbounds for ix in 0:N-1, iy in 0:N-1, iz in 0:N-1
        kx = kf * (ix <= N÷2 ? ix : ix-N)
        ky = kf * (iy <= N÷2 ? iy : iy-N)
        kz = kf * (iz <= N÷2 ? iz : iz-N)
        
        k_mag = sqrt(kx^2 + ky^2 + kz^2)
        if k_mag == 0.0; continue; end
        
        amp = amp_factor * sqrt(P_interp(k_mag))
        δk[ix+1, iy+1, iz+1] = amp * (randn() + im*randn())
    end

    return real.(ifft(δk))
end

# =====================================================================
# 2. ODE SOLVER (JUSILUN)
# =====================================================================
@inline function derivatives(ρ, Θ, Σ, W, V, Λ)
    @fastmath begin
        dρ = -ρ*Θ
        dΘ = -(1/3)*Θ^2 - (1/2)*ρ - 6*Σ^2 + Λ
        dΣ = -(2/3)*Θ*Σ + Σ^2 - W
        dW = -Θ*W - (1/2)*ρ*Σ - 3*Σ*W
        dV = V*Θ
        return dρ, dΘ, dΣ, dW, dV
    end
end

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

function get_H_avg_rel(Θ_vec, V_vec, H_0_c)
    sum_TV = 0.0
    sum_V = 0.0
    @inbounds @simd for i in eachindex(Θ_vec, V_vec)
        sum_TV += Θ_vec[i] * V_vec[i]
        sum_V += V_vec[i]
    end
    return (sum_TV / sum_V) / 3.0 / H_0_c
end

function rk4_step!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec, active_indices, dt, Λ)
    @inbounds for i in active_indices
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

        if Θ_vec[i] < 0.0
            Θ_vec[i] = 0.0
        end
    end
end

function evolve!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec, Λ, H_0_c)
    active_indices = collect(1:length(ρ_vec))
    H_avg_rel = Θ_vec[end]/3 / H_0_c  #get_H_avg_rel(Θ_vec, V_vec, H_0_c)
    step = 0

    while H_avg_rel >= 1.0 && !isempty(active_indices)
        dt = get_dt(Θ_vec, Σ_vec, active_indices)
        rk4_step!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec, active_indices, dt, Λ)
        filter!(i -> Θ_vec[i] > 0.0, active_indices)
        H_avg_rel = Θ_vec[end]/3 / H_0_c  #get_H_avg_rel(Θ_vec, V_vec, H_0_c)

        if step % 250 == 0
            println("Step $step: H_avg/H_0 = $(round(H_avg_rel, digits=4)) | Active = $(length(active_indices)) | dt = $(round(dt, digits=6))")
        end
        step += 1
    end
    H_avg_rel = get_H_avg_rel(Θ_vec, V_vec, H_0_c)
    println("Finished at step $step with H_avg=$(round(H_avg_rel, digits=4))H_0")
end


function jusilun(g_size, h, Ω_m, Ω_Λ, z_i, δ_input, run_id_str)
    H_0 = h * 1.0227e-1 # Convert H_0 from km/s/Mpc to 1/Gyr
    Λ   = 3 * Ω_Λ * H_0^2 / c^2
    ρ_0 = 3 * Ω_m * H_0^2 / c^2

    δ_i = vcat(vec(δ_input), 0.0) 

    ρ_bg_i = ρ_0 * (1 + z_i)^3
    Θ_bg_i = 3 * (H_0/c) * sqrt(Ω_m * (1 + z_i)^3 + Ω_Λ + (1 - Ω_m - Ω_Λ) * (1 + z_i)^2)

    ρ_vec = ρ_bg_i .* (1 .+ δ_i)
    Θ_vec = Θ_bg_i .* (1 .- δ_i ./ 3)
    Σ_vec = Θ_bg_i .* δ_i ./ 9
    W_vec = .-(ρ_bg_i .* δ_i) ./ 6
    V_vec = ones(length(ρ_vec))  

    H_0_c = H_0 / c 
    evolve!(ρ_vec, Θ_vec, Σ_vec, W_vec, V_vec, Λ, H_0_c)
    
    mkpath("./data")
    filename_final = "./data/final_" * run_id_str * ".npz"
    npzwrite(filename_final, Dict(
        "rho" => ρ_vec, "theta" => Θ_vec, "sigma" => Σ_vec, "W" => W_vec, "V" => V_vec,
        "H_0" => H_0, "Omega_m" => Ω_m, "Omega_Lambda" => Ω_Λ, "z_i" => z_i
    ))
    println("Saved final data to: $filename_final")
end

# =====================================================================
# 3. END-TO-END TEST
# =====================================================================
function run_single_test()
    println("--- Starting End-to-End Test ---")
    
    N = 64
    Lbox = 256.0
    zstart = 90.0
    seed = 42
    
    Om = 0.30
    OL = 0.70
    Ob = 0.05
    h = 0.70
    run_id_str = "test_end2end_01"
    
    println("1. Generating Initial Conditions...")
    delta_array = get_density_field(Om, Ob, OL, h, seed, run_id_str, N=N, Lbox=Lbox, zstart=zstart)
    
    println("2. Running Jusilun ODE Solver...")
    @time jusilun(N, h, Om, OL, zstart, delta_array, run_id_str)
end

run_single_test()


density_data = npzread("./data/final_test_end2end_01.npz")["rho"]
reshaped_density = reshape(density_data[1:end-1], (64, 64, 64))
density_bg = density_data[end]  # The last element is the background density
using Plots
heatmap(reshaped_density[:, :, 32]/density_bg, title="Density Slice at z=0", xlabel="X-axis", ylabel="Y-axis", aspect_ratio=1)

