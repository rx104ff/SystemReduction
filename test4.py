import numpy as np
from scipy.linalg import orth

def random_positive_diagonal_matrix(n, low=100, high=1000):
    # Generate a random vector of values in the range [low, high]
    random_values = np.random.uniform(low, high, n)
    # Create a diagonal matrix using the random values
    diagonal_matrix = np.diag(random_values)
    return diagonal_matrix


# Parameters
n = 5   # Dimension of the original space
r = 3   # Dimension of the projected subspace

# Generate a random orthogonal matrix S of size (n x n)
# Start with a random matrix, then use QR decomposition to make it orthogonal
random_matrix = np.random.randn(n, n)
Q, _ = np.linalg.qr(random_matrix)
S = Q  # Use Q as our orthogonal matrix S

# Take the first r columns of S to form S_{(1:r)}
S_1_r = S[:, :r]

# Compute the projection matrix P = S_{(1:r)} * S_{(1:r)}^T
P = S_1_r @ S_1_r.T

# Vector to test the projection
one_vector = np.ones((n, 1))

# Apply the projection matrix to the vector
projected_vector = P @ one_vector

R = random_positive_diagonal_matrix(n)

# Display results
print("Orthogonal Matrix S:\n", S)
print("\nSubmatrix S_{(1:r)}:\n", S_1_r)
print("\nProjection Matrix P = S_{(1:r)} * S_{(1:r)}^T:\n", P)
print("\nOriginal Vector (1s):\n", one_vector.flatten())
print("\nProjected Vector:\n", projected_vector.flatten())
print("\nIs the projection close to the original vector?", np.allclose(projected_vector, one_vector))
