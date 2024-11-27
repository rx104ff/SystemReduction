import numpy as np
from scipy.linalg import null_space
c = np.array([3,3])
A = np.array([[3, -1],[-1, 4]])
B = np.array([[1, 2, -1],[0, -1, 4]])

x = np.array([1,2,3,4,3,2,2])
outer_product = np.outer(x, x)

T = np.array([[1,0,0], [1,0,0], [0,1,0], [0,1,0], [0,1,0],[0,0,1],[0,0,1]])


# Perform eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(outer_product)

# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices for sorting in descending order
eigenvalues = eigenvalues[sorted_indices]        # Sort eigenvalues
eigenvectors = eigenvectors[:, sorted_indices]   # Sort eigenvectors accordingly

# Create the diagonal matrix of sorted eigenvalues (Lambda)
Lambda = np.diag(eigenvalues)

# Reconstruct the matrix using U * Lambda * U^T
reconstructed_matrix = eigenvectors @ Lambda @ eigenvectors.T

kernel_basis = null_space((eigenvectors.T @ T).T)

# Number of basis vectors in the kernel
num_basis_vectors = kernel_basis.shape[1]

# Randomly assign integer coefficients for all but the last basis vector
random_coeffs = np.random.randint(1, 10, size=num_basis_vectors - 1)

# Calculate the coefficient for the last basis vector to ensure y[0] = 0
# Sum up the contributions of the first components from the chosen random coefficients
target_sum = -np.sum(kernel_basis[0, :-1] * random_coeffs)

# Determine the coefficient for the last basis vector
if kernel_basis[0, -1] != 0:
    last_coeff = target_sum / kernel_basis[0, -1]
else:
    last_coeff = 0  # In case the first component of the last basis vector is zero

# Combine all coefficients
coeffs = np.append(random_coeffs, last_coeff)

# Construct y as a linear combination of the kernel basis vectors
y = (kernel_basis @ coeffs).T
y_T = y.reshape([-1, 1])

a = y @ reconstructed_matrix @ y.T

# print("Reconstructed matrix (U * Lambda * U^T):\n", reconstructed_matrix)
print(kernel_basis)
print(T)
print(a)
