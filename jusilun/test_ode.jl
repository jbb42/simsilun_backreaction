using DelimitedFiles, FFTW, Random, Interpolations, NPZ, DifferentialEquations, LinearAlgebra

ENV["OMP_NUM_THREADS"] = "1"
const class_exec = "./class_public-3.3.4/class"

# Constants: Mpc / M_sun / Gyr unit system
const G = 4.498234911e-15
const c = 306.5926758

# =====================================================================
# 1. POWER SPECTRUM FROM CLASS
# =====================================================================
function run_class(Om, Ob, OL, h, run_id; kmax, zstart)
    ini  = "class_$(run_id).ini"
    root = "class_out_$(run_id)_"
    open(ini, "w") do io
        print(io, """
            output = mPk
            z_pk = $zstart
            P_k_max_h/Mpc = $kmax
            Omega_cdm = $(Om - Ob)
            Omega_b = $Ob
            Omega_Lambda = $OL
            h = $h
            n_s = 0.966
            sigma8 = 0.81
            root = $root
        """)
    end
    run(`$class_exec $ini`)
    data = readdlm("$(root)00_pk.dat", comments=true, comment_char='#')
    foreach(f -> rm(f, force=true),
            [ini, "$(root)parameters.ini", ["$(root)0$(i)_pk.dat" for i in 0:5]...])
    return LinearInterpolation(data[:,1], data[:,2], extrapolation_bc=0.0)
end

# =====================================================================
# 2. GAUSSIAN RANDOM FIELD
# =====================================================================
function get_density_field(Om, Ob, OL, h, seed, run_id; N=64, Lbox=256.0, zstart=90.0)
    P = run_class(Om, Ob, OL, h, run_id; kmax=2π*N/Lbox, zstart)

    Random.seed!(seed)
    kf  = 2π / Lbox
    amp = N^3 / sqrt(2 * Lbox^3)
    δk  = zeros(ComplexF64, N, N, N)

    for ix in 0:N-1, iy in 0:N-1, iz in 0:N-1
        kx = kf * (ix <= N÷2 ? ix : ix - N)
        ky = kf * (iy <= N÷2 ? iy : iy - N)
        kz = kf * (iz <= N÷2 ? iz : iz - N)
        k  = sqrt(kx^2 + ky^2 + kz^2)
        k == 0 && continue
        δk[ix+1, iy+1, iz+1] = amp * sqrt(P(k)) * (randn() + im*randn())
    end

    return real.(ifft(δk))
end

# =====================================================================
# 3. SINGLE-CELL ODE  (state: [ρ, Θ, Σ, W, V],  param: Λ)
# =====================================================================
function cell_ode!(du, u, Λ, t)
    ρ, Θ, Σ, W, V = u
    du[1] = -ρ*Θ
    du[2] = -(Θ^2)/3 - ρ/2 - 6*Σ^2 + Λ
    du[3] = -(2*Θ*Σ)/3 + Σ^2 - W
    du[4] = -Θ*W - ρ*Σ/2 - 3*Σ*W
    du[5] = V*Θ
end

# =====================================================================
# 4. FIND t_end: time when background H drops to H₀
# =====================================================================
function find_t_end(ρ_bg, Θ_bg, Λ, H₀_c)
    # Background (δ=0) has Σ=W=0 forever, so only ρ and Θ evolve
    function bg_ode!(du, u, Λ, t)
        ρ, Θ = u
        du[1] = -ρ*Θ
        du[2] = -(Θ^2)/3 - ρ/2 + Λ
    end
    cb = ContinuousCallback(
        (u, t, integ) -> u[2] - 3*H₀_c,
        nothing, terminate!;
        rootfind=SciMLBase.RightRootFind, save_positions=(false, true))
    sol = solve(ODEProblem(bg_ode!, [ρ_bg, Θ_bg], (0.0, 1e4), Λ), Tsit5();
                callback=cb, reltol=1e-12, abstol=1e-14, verbose=false)
    sol.retcode == ReturnCode.Terminated ||
        error("Background never reached H=H₀  (retcode: $(sol.retcode))")
    println("Background reaches H=H₀ at t = $(round(sol.t[end], digits=4)) Gyr")
    return sol.t[end]
end

# =====================================================================
# 5. MAIN: SET UP ICs, EVOLVE ALL CELLS, SAVE
# =====================================================================
function jusilun(h, Ω_m, Ω_Λ, z_i, δ, run_id)
    H₀    = h * 1.0227e-1      # km/s/Mpc → 1/Gyr
    H₀_c  = H₀ / c
    Λ     = 3Ω_Λ * H₀^2 / c^2

    # Initial conditions from linear perturbation theory.
    # Stored as Vector{SVector{5}} — prob_func indexes directly with zero allocation.
    δ_cells = vcat(vec(δ), 0.0)  # append unperturbed background cell
    ρ_bg    = 3Ω_m * H₀^2 / c^2 * (1 + z_i)^3
    Θ_bg    = 3H₀_c * sqrt(Ω_m*(1+z_i)^3 + Ω_Λ + (1-Ω_m-Ω_Λ)*(1+z_i)^2)

    u0 = [Vector{Float64}([ρ_bg*(1+d), Θ_bg*(1-d/3), Θ_bg*d/9, -ρ_bg*d/6, 1.0])
          for d in δ_cells]

    t_end = find_t_end(u0[end][1], u0[end][2], Λ, H₀_c)

    # cb_collapse is local so re-running jusilun never hits the const redefinition error
    cb_collapse = ContinuousCallback(
        (u, t, integ) -> u[2],
        nothing, terminate!;
        rootfind=SciMLBase.RightRootFind, save_positions=(false, true))

    # EnsembleSerial: outer universe loop will parallelise with Threads.@threads
    base_prob = ODEProblem(cell_ode!, zeros(5), (0.0, t_end), Λ; callback=cb_collapse)
    ensemble  = EnsembleProblem(base_prob;
                    prob_func  = (prob, i, _) -> remake(prob; u0=u0[i]),
                    safetycopy = false)
    sim = solve(ensemble, Tsit5(), EnsembleSerial();
                trajectories=length(u0), reltol=1e-8, abstol=1e-10,
                save_everystep=false, verbose=false)

    # Collect final states into a preallocated matrix (no mapreduce temporaries)
    final = Matrix{Float64}(undef, length(u0), 5)
    for (i, s) in enumerate(sim.u)
        final[i, :] .= s.u[end]
    end
    final[:,2] .= max.(final[:,2], 0.0)  # clamp tiny negative Θ at readout

    n_cells     = length(u0) - 1  # exclude background
    n_collapsed = count(s -> s.retcode == ReturnCode.Terminated, sim.u) - 1
    H_avg       = dot(final[:,2], final[:,5]) / sum(final[:,5]) / 3 / H₀_c
    println("H_avg/H₀ = $(round(H_avg, digits=4)) | Collapsed: $n_collapsed / $n_cells")

    mkpath("./data")
    npzwrite("./data/final_$(run_id).npz", Dict(
        "rho"=>final[:,1], "theta"=>final[:,2], "sigma"=>final[:,3],
        "W"  =>final[:,4], "V"    =>final[:,5],
        "H_0"=>H₀, "Omega_m"=>Ω_m, "Omega_Lambda"=>Ω_Λ, "z_i"=>z_i))
    println("Saved: ./data/final_$(run_id).npz")
    return final
end

# =====================================================================
# 6. RUN + PLOT
# =====================================================================
run_id = "test_end2end_01"
δ      = get_density_field(0.30, 0.05, 0.70, 0.70, 42, run_id; N=64, Lbox=256.0, zstart=90.0)
@time final = jusilun(0.70, 0.30, 0.70, 90.0, δ, run_id)

using Plots
ρ_bg    = final[end, 1]
ρ_slice = reshape(final[1:end-1, 1], 64, 64, 64)[:, :, 32]
heatmap(ρ_slice ./ ρ_bg; title="Density / ρ_bg  (z=0)", xlabel="X", ylabel="Y", aspect_ratio=1)