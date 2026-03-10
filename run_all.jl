using Base.Threads
using Random
using Printf

include("jusilun.jl")

h_vals = range(0.6, 0.8, length=2)
Ωm_vals = range(0.2, 0.4, length=2)
ΩΛ_vals = range(0.6, 0.8, length=2) # 21 steps
runs_per_param = 0:0 # Up to 9

# Create the flattened job list
all_jobs = collect(Iterators.product(h_vals, Ωm_vals, ΩΛ_vals, runs_per_param))

println("Starting parameter sweep with $(length(all_jobs)) total runs...")

# Thread over the independent runs
Threads.@threads for (h, Ωm, ΩΛ, run_idx) in all_jobs
    Ωk = 1.0 - Ωm - ΩΛ 
    
    # Use run_idx to ensure unique file names for CLASS and the output
    run_id = @sprintf("h%.2d_m%.2d_L%.2d_run%02d", h*100, Ωm*100, ΩΛ*100, run_idx)    
    if isfile("./output_data/$(run_id).npz")
        println("Skipping $run_id (already exists)")
        continue
    end

    println("Thread $(Threads.threadid()) starting: $run_id")
    
    # Run the simulation
    jusilun(Ωm, ΩΛ, Ωk, h, rand(UInt32), run_id; N=64, Lbox=256.0, zi=90.0, headless=true)
end

println("All runs completed successfully!")