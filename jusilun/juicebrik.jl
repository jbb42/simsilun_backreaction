using NPZ
using Base.Threads
println("nthreads() = ", nthreads())

function jusilun(g_size, H_0_km_s_Mpc, Ω_m, Ω_Λ, z_i)
    # Simsilun units
    mu = 1.989e45   # 10^15 solar masses
    lu = 3.085678e19    # 10 kpc
    tu = 31557600.0 * 1e6   # 1 mega years

    G   = 6.6742e-11 * mu * tu^2 / lu^3
    c   = 299792458.0 * tu / lu
    κ   = 8*pi*G/c^4
    # Geometric units
    H_0 = tu / lu * H_0_km_s_Mpc
    Λ   = 3 * Ω_Λ * H_0^2 / c^2
    ρ_0 = 3 * Ω_m * H_0^2 / c^2

    # Initial conditions
    δ_i = npzread("./data/ics/delta.npy")
    δ_i = reshape(δ_i, g_size^3)
    push!(δ_i, 0.0) # add one more point for the background
    ρ_bg_i = ρ_0 * (1 + z_i)^3
    Θ_bg_i = 3 * (H_0/c) * sqrt(Ω_m * (1 + z_i)^3 + Ω_Λ + (1 - Ω_m - Ω_Λ) * (1 + z_i)^2)

    ρ_vec = ρ_bg_i .* (1 .+ δ_i)

    # DELETE BLOCK
    basepath = "./data/jusilun_output/initial_vals"
    i = 0
    filename = basepath * "_" * lpad(string(i), 3, '0') * ".npz"
    while isfile(filename)
        i += 1
        filename = basepath * "_" * lpad(string(i), 3, '0') * ".npz"
    end
    npzwrite(filename, Dict(
        "rho" => ρ_vec))
end