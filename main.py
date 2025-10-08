from initial_conditions.generate_ic import *
from initial_conditions.read_params import *
# call_julia.py
import os
os.environ["JULIA_NUM_THREADS"] = "8"  # number of threads you want
from julia import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.include("jusilun/jusilun.jl")

filename = "./initial_conditions/ngenic.param"  # path to your .param file
new_values = {
    "HubbleParam": 0.70,
    "Omega": 0.31,
    "OmegaLambda": 0.7,
    "Redshift": 90,
    "Nmesh": 64,
}

update_params(filename, new_values)
params = read_params(filename)

class_dict = {'h': params['HubbleParam'],
               'Omega_m': params['Omega'],
               'Omega_Lambda': params['OmegaLambda'],
               'Omega_k': 1-params['Omega']-params['OmegaLambda'],
               'A_s': 2.1e-9,
               'n_s': 0.965,
               'tau_reio' :0.054,
               'z_pk': params['Redshift'],
               'output': 'mPk,dTk' ,
               'P_k_max_h/Mpc': 100,  # extend upper range if needed
           }

run_class(class_dict, params['Redshift'])
dens_contrast(params['Nmesh'])

# Call julia
Main.jusilun(params['Nmesh'],
             params['HubbleParam']*100,
             params['Omega'],
             params['OmegaLambda'],
             params['Redshift'])

from plot import *
plot_grid(params['Nmesh'])
