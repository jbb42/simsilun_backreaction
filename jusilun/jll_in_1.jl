import Pkg
using NPZ
using FFTW
using Random
using DelimitedFiles
using Interpolations
using LinearAlgebra
using Printf
using Base.Threads

#====================================================================#
# Defining constants in simsilun units
#====================================================================#

# Force CLASS to only use 1 thread to prevent oversubscription
ENV["OMP_NUM_THREADS"] = "1"

# Constants in cosmological units
const G = 4.498234911e-15  # G in Mpc^3/(M_sun*Gyr^2)
const c = 306.5926758      # Speed of light [Mpc/Gyr]
const κ = 8*pi*G/c^4

# Ensure output directories exist
mkpath("./data/jusilun_output")
mkpath("./initial_conditions")

#====================================================================#
# Generate initial conditions using CLASS
#====================================================================#

function run_class(Ωm, ΩΛ, Ωk, h, seed, id; N=64, Lbox=256.0, zi=90.0)
    # Set wavenumber based on Nyquist frequency
    kNy = π * N / Lbox
    kmax = 2.0 * kNy

    # Write CLASS input file
    class_ini = "./initial_conditions/class_$(id).ini"
    class_out = "./initial_conditions/class_out_$(id)_"

    open(class_ini, "w") do io
        println(io, "output = mPk")
        println(io, "z_pk = $zi")
        println(io, "P_k_max_h/Mpc = $kmax")
        println(io, "Omega_m = $Ωm")
        println(io, "Omega_Lambda = $ΩΛ")
        println(io, "Omega_k = $Ωk")
        println(io, "h = $h")
        println(io, "n_s = 0.966")
        println(io, "sigma8 = 0.81")
        println(io, "root = $class_out")
    end

    # Run CLASS and read power spectrum
    run(`./class_public-3.3.4/class $class_ini`)
    ps = readdlm("$(class_out)00_pk.dat", comments=true, comment_char='#')

    # Clean up CLASS output files
    foreach(f -> rm(f, force=true),
            [class_ini, "$(class_out)parameters.ini", ["$(class_out)0$(i)_pk.dat" for i in 0:5]...])

    # Return interpolated power spectrum
    return linear_interpolation(ps[:,1], ps[:,2], extrapolation_bc=0.0)
end

#====================================================================#
# Generate initial conditions using CLASS
#====================================================================#