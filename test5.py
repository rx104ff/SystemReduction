import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals

# Define matrix A (symmetric, positive, and Metzler - negative definite)
n = 5  # Dimension of the matrix
np.random.seed(0)  # For reproducibility

# Create a negative definite symmetric matrix A by ensuring all eigenvalues are negative
D = -np.diag(np.random.uniform(1, 2, n))  # Diagonal with negative entries
M = np.random.rand(n, n)
M = (M + M.T) / 2  # Symmetric matrix
A = D - 0.5 * M  # Metzler matrix: off-diagonal elements non-negative

# Ensure all off-diagonal elements are non-negative
A = np.where(np.eye(n) == 1, A, np.abs(A))

# Define vector b with non-negative elements
b = np.random.rand(n)
b[b < 0] = 0  # Ensure all elements are non-negative

# Define range of w values
w_values = np.linspace(0, 10, 100)
largest_eigenvalues = []

# Experiment with each value of w
for w in w_values:
    s = 1j * w  # s = jw
    # Compute v = (sI - A)^(-1) * b
    try:
        v = np.linalg.solve(s * np.eye(n) - A, b)
        # Compute vv^* (outer product of v with its conjugate transpose)
        vv_star = np.outer(v, np.conjugate(v))
        # Compute the largest eigenvalue of vv^*
        print(sorted(eigvals(vv_star).real))
        max_eigenvalue = np.max(eigvals(vv_star).real)  # Only take real part
        largest_eigenvalues.append(max_eigenvalue)
    except np.linalg.LinAlgError:
        largest_eigenvalues.append(np.nan)  # Handle singular matrix cases

# Plot largest eigenvalue vs w
plt.figure(figsize=(10, 6))
plt.plot(w_values, largest_eigenvalues, label=r'$\lambda_{\max}$ of $vv^*$')
plt.xlabel(r'$w$')
plt.ylabel(r'$\lambda_{\max}$')
plt.title('Largest Eigenvalue of $vv^*$ as a Function of $w$')
plt.legend()
plt.grid()
plt.show()
