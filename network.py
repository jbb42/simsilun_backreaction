import numpy as np

N = 1

oms = [0.20 + 0.02 * i for i in range(11)]
ols = [0.60 + 0.02 * i for i in range(11)]

#oms = [0.20]
#ols = [0.60 + 0.02 * i for i in range(2)]

features = []
labels = []

for n in range(N):
    for om in oms:
        for ol in ols:
            filename = f"data/jusilun_output/i_m{om:.2f}_L{ol:.2f}_n{n:02d}.npz"
            data = np.load(filename)

            feature = data["rho"]
            label = 1 - data["Omega_m"] - data["Omega_Lambda"]

            features.append(feature)
            labels.append(label)

print(np.array(features))
print(np.array(labels))


# Network
from tensorflow import keras
from sklearn.model_selection import train_test_split
