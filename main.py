from initial_conditions.generate_ic import *
from initial_conditions.read_params import *
import os
os.environ["JULIA_NUM_THREADS"] = "16"  # number of threads you want
from julia import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.include("jusilun/jusilun.jl")

N_grid = 64
filename = "./initial_conditions/ngenic.param"  # path to your .param file

#oms = [0.2 + 0.02 * i for i in range(11)]
#ols = [0.6 + 0.02 * i for i in range(11)]

oms = [0.3]
ols = [0.7]
for _ in range(1):
#for _ in range(10):
    for om in oms:
        for ol in ols:
            new_values = dict(HubbleParam=0.70,
                              Omega=om,
                              OmegaLambda=ol,
                              Redshift=90,
                              Nmesh=N_grid,
                              NBaryon=N_grid,
                              NCDM=N_grid,
                              Seed=np.random.randint(1_000_000))
            print("Seed = ", new_values["Seed"])
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
                           'P_k_max_h/Mpc': 200,  # extend upper range if needed
                       }

            run_class(class_dict, params['Redshift'])
            dens_contrast(params['Nmesh'])
            #    delta = np.load("/home/jbb/Downloads/delta.npy")
            #    print(delta.shape)
            #    N = 256
            #    block = 4
            #    new_N = N // block
            #    delta = delta.reshape(new_N, block, new_N, block, new_N, block).mean(axis=(1, 3, 5))*64
            #    print(delta)
            #    np.save("./data/ics/delta.npy", delta)

            # Call julia
            Main.jusilun(params['Nmesh'],
                         params['HubbleParam']*100,
                         params['Omega'],
                         params['OmegaLambda'],
                         params['Redshift'])

from plot import *
plot_grid(params['Nmesh'])
