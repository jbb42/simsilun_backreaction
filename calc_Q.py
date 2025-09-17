import numpy as np
from matplotlib import pyplot as plt

def radial_cumsum(arr_3d, center):
    """
    Calculates the cumulative sum of elements in a 3D array, sorted by radial distance from the center.

    Args:
        arr_3d (np.ndarray): The input 3D NumPy array.
        center (tuple): A tuple of three integers representing the coordinates (z, y, x) of the center.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A 1D NumPy array containing the cumulative sum of elements sorted by radial distance.
            - np.ndarray: A 1D NumPy array containing the original indices of the sorted elements.
    """
    z_dim, y_dim, x_dim = arr_3d.shape
    z_center, y_center, x_center = center

    # Create coordinate grids
    z_coords, y_coords, x_coords = np.indices(arr_3d.shape)

    # Calculate radial distance from the center
    radial_distances = np.sqrt((z_coords - z_center)**2 + (y_coords - y_center)**2 + (x_coords - x_center)**2)

    # Flatten the array and radial distances
    flat_arr = arr_3d.ravel()
    flat_radial_distances = radial_distances.ravel()

    # Sort indices based on radial distance
    sorted_indices = np.argsort(flat_radial_distances)

    # Sort the flattened array based on sorted indices
    sorted_arr = flat_arr[sorted_indices]

    # Calculate the cumulative sum
    cumulative_sum = np.cumsum(sorted_arr)

    return cumulative_sum, sorted_indices

def reshape_radial_cumsum(original_shape, cumulative_sum, sorted_indices):
    """
    Reshapes a flattened cumulative sum array back to its original 3D shape
    using the provided sorted indices.

    Args:
        original_shape (tuple): A tuple representing the original shape (z, y, x) of the array.
        cumulative_sum (np.ndarray): A 1D NumPy array containing the cumulative sum values
                                     sorted by radial distance.
        sorted_indices (np.ndarray): A 1D NumPy array containing the original indices
                                     of the sorted elements.

    Returns:
        np.ndarray: A 3D NumPy array with the cumulative sum values in their original
                    spatial arrangement.
    """
    # Create an empty array with the original shape
    reshaped_array = np.empty(original_shape)

    # Flatten the empty array to match the flattened cumulative_sum and sorted_indices
    flat_reshaped_array = reshaped_array.ravel()

    # Place the cumulative_sum values back into their original positions using sorted_indices
    flat_reshaped_array[sorted_indices] = cumulative_sum

    # Reshape the flattened array back to the original shape
    reshaped_array = flat_reshaped_array.reshape(original_shape)

    return reshaped_array



H_0=70
z_i=1100.0
z_f=0.0
mu=1.989e45
lu=3.085678e19
tu=31557600e6
G = 6.6742e-11*((mu*(tu**2))/(lu**3))
c = 299792458*(tu/lu)
H0 = (tu / lu) * H_0
t_0 = (2 / 3) * (1 / H0)  # Present time
a_i = 1 / (1 + z_i)  # Initial scale factor
t_i = t_0 * a_i ** (3 / 2)  # Initial time
a_f = 1 / (1 + z_f)  # Final scale factor
t_f = t_0 * a_f ** (3 / 2)  # Final time
t = np.array([t_i, t_f])

def rho_eds(i):
    return 3 * c ** 2 * ((2 / (3 * t[i])) ** 2 / (8 * np.pi * G))

def a_eds(i):
    return (t[i] / t_0) ** (2 / 3)

def a_eds_dt(i):
  return 2/3*(t[i])**(-1/3)*(t_0)**(-2/3)

def theta_eds(i):
  return 3*a_eds_dt(i)/a_eds(i)

# Load data
rho_f = np.load("data/grid_rho1.npy")*rho_eds(1)
theta_f = np.load("data/grid_theta1.npy")*theta_eds(1)
sigma_f = np.load("data/grid_sigma1.npy")*theta_eds(1)/3
weyl_f = np.load("data/grid_weyl1.npy")
V_f = np.load("data/grid_V1.npy")

rho_i = np.load("data/grid_rho0.npy")*rho_eds(0)
theta_i = np.load("data/grid_theta0.npy")*theta_eds(0)
sigma_i = np.load("data/grid_sigma0.npy")*theta_eds(0)/3
weyl_i = np.load("data/grid_weyl0.npy")
V_i = np.load("data/grid_V0.npy")

# Assuming the center is the middle of the array
center_f = tuple(np.array(theta_f.shape) // 2)
center_i = tuple(np.array(theta_i.shape) // 2)

# Calculate radial cumulative sums and sorted indices for components of Q_f
theta_Vf_cumsum, sorted_indices_f = radial_cumsum(theta_f * V_f, center_f)
Vf_cumsum, _ = radial_cumsum(V_f, center_f)
theta2_Vf_cumsum, _ = radial_cumsum(theta_f**2 * V_f, center_f)
sigma2_Vf_cumsum, _ = radial_cumsum(sigma_f**2 * V_f, center_f)
sigma_Vf_cumsum, _ = radial_cumsum(sigma_f * V_f, center_f)


# Calculate Q_f based on radial cumulative sums
theta_Qf_radial = theta_Vf_cumsum / Vf_cumsum
sigma_Qf_radial = sigma_Vf_cumsum / Vf_cumsum
theta2_Qf_radial = theta2_Vf_cumsum / Vf_cumsum
sigma2_Qf_radial = sigma2_Vf_cumsum / Vf_cumsum

Q_f_radial = 2/3*(theta2_Qf_radial-theta_Qf_radial**2)-6*sigma2_Qf_radial

# Reshape Q_f_radial back to original shape
original_shape_f = theta_f.shape
Q_f_reshaped = reshape_radial_cumsum(original_shape_f, Q_f_radial, sorted_indices_f)


# Calculate radial cumulative sums and sorted indices for components of Q_i
theta_Vi_cumsum, sorted_indices_i = radial_cumsum(theta_i * V_i, center_i)
Vi_cumsum, _ = radial_cumsum(V_i, center_i)
theta2_Vi_cumsum, _ = radial_cumsum(theta_i**2 * V_i, center_i)
sigma2_Vi_cumsum, _ = radial_cumsum(sigma_i**2 * V_i, center_i)
sigma_Vi_cumsum, _ = radial_cumsum(sigma_i * V_i, center_i)

# Calculate Q_i based on radial cumulative sums
theta_Qi_radial = theta_Vi_cumsum / Vi_cumsum
sigma_Qi_radial = sigma_Vi_cumsum / Vi_cumsum
theta2_Qi_radial = theta2_Vi_cumsum / Vi_cumsum
sigma2_Qi_radial = sigma2_Vi_cumsum / Vi_cumsum

Q_i_radial = 2/3*(theta2_Qi_radial-theta_Qi_radial**2)-6*sigma2_Qi_radial

# Reshape Q_i_radial back to original shape
original_shape_i = theta_i.shape
Q_i_reshaped = reshape_radial_cumsum(original_shape_i, Q_i_radial, sorted_indices_i)

plt.imshow(Q_f_reshaped[:,:,32])
plt.colorbar()
plt.title("Q_f")
plt.show()

plt.imshow(Q_i_reshaped[:,:,32])
plt.colorbar()
plt.title("Q_i")
plt.show()


plt.imshow(np.log(Q_f_reshaped[:,:,32]))
plt.colorbar()
plt.title("log(Q_f)")
plt.show()

plt.imshow(np.log(Q_i_reshaped[:,:,32]))
plt.colorbar()
plt.title("log(Q_i)")
plt.show()