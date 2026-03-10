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
# 2. GAUSSIAN RANDOM FIELD  (irfft half-grid)
# =====================================================================
function get_density_field(Om, Ob, OL, h, seed, run_id; N=64, Lbox=256.0, zstart=90.0)
    P       = run_class(Om, Ob, OL, h, run_id; kmax=2π*N/Lbox, zstart)
    kf      = 2π / Lbox
    amp     = N^3 / sqrt(2.0 * Lbox^3)
    Nx_half = N ÷ 2 + 1
    δk      = zeros(ComplexF64, Nx_half, N, N)

    Random.seed!(seed)
    @inbounds for ix in 0:Nx_half-1, iy in 0:N-1, iz in 0:N-1
        kx = kf * ix
        ky = kf * (iy <= N÷2 ? iy : iy - N)
        kz = kf * (iz <= N÷2 ? iz : iz - N)
        k  = sqrt(kx^2 + ky^2 + kz^2)
        k == 0 && continue
        δk[ix+1, iy+1, iz+1] = amp * sqrt(P(k)) * (randn() + im*randn())
    end
    return irfft(δk, N)
end

# =====================================================================
# 3. ODES
# =====================================================================
function bg_ode!(du, u, Λ, t)
    ρ, Θ = u[1], u[2]
    du[1] = -ρ*Θ
    du[2] = -(Θ^2)/3 - ρ/2 + Λ
end

function grid_ode!(du, u, p, t)
    Λ, active = p.Λ, p.active
    @inbounds for i in axes(u, 2)
        if active[i]
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

# =====================================================================
# 4. COLLAPSE CALLBACK  (discrete + linear interpolation to zero-crossing)
# =====================================================================
function collapse_condition(u, t, integrator)
    active = integrator.p.active
    @inbounds for i in axes(u, 2)
        if active[i] && u[2, i] <= 0.0
            return true
        end
    end
    return false
end

function collapse_affect!(integrator)
    active = integrator.p.active
    u = integrator.u
    changed = false

    @inbounds for i in axes(u, 2)
        if active[i] && u[2, i] <= 0.0
            active[i] = false 
            
            # Snap expansion rate to exactly zero. 
            # The other variables freeze exactly where they are.
            u[2, i] = 0.0 
            
            changed = true
        end
    end
    
    if changed
        u_modified!(integrator, true)
    end
end

cb_collapse = DiscreteCallback(collapse_condition, collapse_affect!;
                                     save_positions=(false, false))

# =====================================================================
# 5. FIND t_end: exact time background H drops to H₀
# =====================================================================
function find_t_end(ρ_bg, Θ_bg, Λ, H₀_c)
    cb = ContinuousCallback(
        (u, t, integ) -> u[2] - 3*H₀_c,
        nothing, terminate!;
        rootfind=SciMLBase.RightRootFind, save_positions=(false, true))
    sol = solve(ODEProblem(bg_ode!, [ρ_bg, Θ_bg], (0.0, 1e4), Λ), Tsit5();
                callback=cb, reltol=1e-12, abstol=1e-14, verbose=false)
    sol.retcode == ReturnCode.Terminated ||
        error("Background never reached H=H₀ (retcode: $(sol.retcode))")
    println("Background reaches H=H₀ at t = $(round(sol.t[end], digits=4)) Gyr")
    return sol.t[end]
end

# =====================================================================
# 6. MAIN
# =====================================================================
function jusilun(h, Ω_m, Ω_Λ, z_i, δ, run_id)
    H₀   = h * 1.0227e-1
    H₀_c = H₀ / c
    Λ    = 3Ω_Λ * H₀^2 / c^2

    ρ_bg = 3Ω_m * H₀^2 / c^2 * (1 + z_i)^3
    Θ_bg = 3H₀_c * sqrt(Ω_m*(1+z_i)^3 + Ω_Λ + (1-Ω_m-Ω_Λ)*(1+z_i)^2)

    t_end = find_t_end(ρ_bg, Θ_bg, Λ, H₀_c)

    # 5 × N_cells matrix: column-major so each cell's variables are contiguous
    δ_cells = vcat(vec(δ), 0.0)   # last cell is unperturbed background
    N_cells = length(δ_cells)
    u0 = Matrix{Float64}(undef, 5, N_cells)
    @inbounds for (i, d) in enumerate(δ_cells)
        u0[1,i] =  ρ_bg * (1 + d)
        u0[2,i] =  Θ_bg * (1 - d/3)
        u0[3,i] =  Θ_bg * d/9
        u0[4,i] = -ρ_bg * d/6
        u0[5,i] =  1.0
    end

    # active is tracked in p so collapse_affect! can mark cells as frozen
    p   = (Λ=Λ, active=fill(true, N_cells))
    sol = solve(ODEProblem(grid_ode!, u0, (0.0, t_end), p), Tsit5();
                callback=cb_collapse, reltol=1e-8, abstol=1e-10,
                save_everystep=false, dense=false, verbose=false)

    u_end = sol.u[end]
    final = Matrix{Float64}(undef, N_cells, 5)
    @inbounds for i in 1:N_cells
        final[i,1] = u_end[1,i];  final[i,2] = max(u_end[2,i], 0.0)
        final[i,3] = u_end[3,i];  final[i,4] = u_end[4,i]
        final[i,5] = u_end[5,i]
    end

    # Use the active array for an unambiguous collapsed count
    n_cells     = N_cells - 1   # exclude background
    n_collapsed = count(!, p.active[1:n_cells])
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
# 7. RUN + PLOT
# =====================================================================
run_id = "test_end2end_01"
δ      = get_density_field(0.30, 0.05, 0.70, 0.70, 42, run_id; N=128, Lbox=256.0, zstart=90.0)
@time final = jusilun(0.70, 0.30, 0.70, 90.0, δ, run_id)

using Plots
ρ_bg    = final[end, 1]
ρ_slice = reshape(final[1:end-1, 1], 128, 128, 128)[:, :, 64]
heatmap(ρ_slice ./ ρ_bg; title="Density / ρ_bg  (z=0)", xlabel="X", ylabel="Y", aspect_ratio=1)