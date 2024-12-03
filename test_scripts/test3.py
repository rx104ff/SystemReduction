import numpy as np


# Function to generate a random positive diagonal matrix with larger values
def random_positive_diagonal_matrix(n, low=100, high=1000):
    # Generate a random vector of values in the range [low, high]
    random_values = np.random.uniform(low, high, n)
    # Create a diagonal matrix using the random values
    diagonal_matrix = np.diag(random_values)
    return diagonal_matrix


# Define the matrix P
P = np.array([[1, 0],
              [0, 1],
              [1, 1],
              [5, 1]])

# Calculate P^T P
PtP = P.T @ P

# Calculate (P^T P)^{-1}
PtP_inv = np.linalg.inv(PtP)

# Calculate the projection matrix P(P^T P)^{-1}P^T
projection_matrix = P @ PtP_inv @ P.T

# Perform eigenvalue decomposition to find Q and Lambda
eigenvalues, eigenvectors = np.linalg.eigh(projection_matrix)

# Sort eigenvalues and corresponding eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Form the diagonal matrix Lambda
Lambda = np.diag(eigenvalues)

# Verify if Q * Lambda * Q^T equals the projection matrix
reconstructed_matrix = eigenvectors @ Lambda @ eigenvectors.T

R = random_positive_diagonal_matrix(4)
Q_0 = eigenvectors.T[0:2].T
Q_1 = eigenvectors.T[2::].T
# Display results
tt = np.linalg.inv(R) @ Q_0 @ Q_0.T @ R
eee = Q_0 @ Q_0.T
print("Original Projection Matrix:\n", projection_matrix)
print("\nEigenvalues (Lambda):\n", Lambda)
print("\nEigenvectors (Q):\n", eigenvectors)
print("\nReconstructed Matrix (Q * Lambda * Q^T):\n", reconstructed_matrix)
