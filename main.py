from initial_conditions.generate_ic import *
from initial_conditions.read_params import *

# === EDIT THESE VALUES ===
filename = "./initial_conditions/ngenic.param"  # path to your .param file
new_values = {
    "HubbleParam": 0.70,
    "Omega": 0.3,
    "OmegaLambda": 0.7,
    "Redshift": 90,
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

               'output': 'mPk,dTk',
               'P_k_max_h/Mpc': 100,  # extend upper range if needed
           }

run_class(class_dict, params['Redshift'])

dens_contrast(params['Nmesh'])