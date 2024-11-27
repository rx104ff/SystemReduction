import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.linalg import eigh, hessenberg

n = 1000
m = 500


def f(s, input_A, input_B, C):
    epsilon = 1e-5  # Small regularization term to avoid singular matrix error
    try:
        return C @ np.linalg.inv(s * np.eye(input_A.shape[0]) - input_A + epsilon * np.eye(input_A.shape[0])) @ input_B
    except np.linalg.LinAlgError:
        # Handle singular matrix by adding a larger regularization term
        epsilon = 1e-3
        return C @ np.linalg.inv(s * np.eye(input_A.shape[0]) - input_A + epsilon * np.eye(input_A.shape[0])) @ input_B


def forward_system(input_A, Kappa, t):
    A_t = input_A.copy()
    for i in range(input_A.shape[0]):
        for j in range(input_A.shape[1]):
            if i != j:
                A_t[i, j] = (input_A[i, j] * np.exp(t)) / (1 + Kappa[i, j] * (np.exp(t) - 1))
    return A_t


def positive_tridiagonalization(A, b):
    """
    Perform positive tridiagonalization on a symmetric matrix A and vector b
    according to the definition.

    Parameters:
    - A: Input symmetric matrix (n x n).
    - b: Input vector (n x 1).

    Returns:
    - A_hat: Tridiagonal matrix with non-negative off-diagonals.
    - b_hat: Vector with a single non-zero positive value.
    - H: Orthogonal transformation matrix.
    """
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix A must be square."
    assert np.allclose(A, A.T), "Matrix A must be symmetric."
    assert b.shape == (n, 1), "Vector b must be a column vector."

    # Initialize H as an identity matrix
    H = np.eye(n)

    # Iteratively construct tridiagonal form
    for k in range(n - 1):
        # Householder transformation for column k
        x = A[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x)
        v = x - e1
        v = v / np.linalg.norm(v)

        # Construct Householder matrix
        H_k = np.eye(n)
        H_k[k:, k:] -= 2.0 * np.outer(v, v)

        # Apply the transformation
        A = H_k @ A @ H_k.T
        H = H @ H_k

    # Ensure non-negative off-diagonal elements
    for i in range(n - 1):
        if A[i, i + 1] < 0:
            A[i, i + 1] *= -1
            A[i + 1, i] *= -1
            H[:, i + 1] *= -1

    # Transform b
    b_hat = H.T @ b

    # Ensure b_hat has only the first element non-zero
    b_hat_sparse = np.zeros_like(b_hat)
    b_hat_sparse[0] = np.linalg.norm(b_hat)

    return A, b_hat_sparse, H


def optimal_truncation(H, b_tilde, input_k):
    """
    Compute truncated matrices for reduced-order approximation.
    """
    H_k = H[:input_k, :input_k]
    b_k = b_tilde[:input_k]
    return H_k, b_k


if os.path.exists('L_matrix.csv') and os.path.exists('R_matrix.csv') and os.path.exists('B_matrix.csv') and os.path.exists('u_value.csv') and os.path.exists('W_matrix.csv'):
    L = pd.read_csv('L_matrix.csv').values
    R = pd.read_csv('R_matrix.csv').values
    B = pd.read_csv('B_matrix.csv').values
    u = pd.read_csv('u_value.csv').values[0, 0]
    W = pd.read_csv('W_matrix.csv').values
    A = -R + L
else:
    # Generate a random graph with an average degree of approximately 70
    p = 70 / (n - 1)  # Probability to achieve average degree of 70
    G = nx.gnp_random_graph(n, p)
    while not nx.is_connected(G):  # Ensure the graph is connected
        G = nx.gnp_random_graph(n, p)

    # Create adjacency matrix L from the graph
    L = nx.to_numpy_array(G)
    L = L * np.random.uniform(0, 50, (n, n))  # Assign random weights to edges
    L = (L + L.T) / 2  # Symmetrize L
    np.fill_diagonal(L, 8
                     -np.sum(L, axis=1))  # Make diagonal negative sum of the rest of the row to ensure Metzler matrix

    R_1 = np.diag(
        np.random.choice([np.random.uniform(500, 999), np.random.uniform(1000, 9999)],
                         size=m))  # Increase variance by using numbers with different digits

    R_2 = np.diag(
        np.random.choice([np.random.uniform(50, 99), np.random.uniform(500, 999)],
                         size=(n-m)))  # Increase variance by using numbers with different digits

    R = np.block([
        [R_1, np.zeros((R_1.shape[0], R_2.shape[1]))],
        [np.zeros((R_2.shape[0], R_1.shape[1])), R_2]
    ])

    L = L + 0.01

    A = -R + L

    B_1 = np.random.rand(m, 1) * 50
    B_2 = np.random.rand(n-m, 1) * 10
    B = np.vstack((B_1, B_2))

    u = 1  # Increase mean and variance for larger value
    x = -np.linalg.solve(A, B * u)

    # Compute W(i, j) using a more efficient approach
    W = np.zeros((n, n))
    non_zero_indices = [np.nonzero(A[i, :])[0] for i in
                        range(n)]

    sum_n = 0
    for x_i in x:
        sum_n += 1/x_i

    for i in range(n):
        for j in range(n):
            if i != j:
                """
                neighbors_i = non_zero_indices[i]
                neighbors_j = non_zero_indices[j]
                sum_neighbors = 0
                for u in neighbors_i:
                    if A[u, i] != 0:
                        sum_neighbors += (1 / x[i])

                for v in neighbors_j:
                    if A[v, j] != 0:
                        sum_neighbors += (1 / x[j])
                """
                W[i, j] = sum_n

    # Save L, R, B, u, W to CSV files
    pd.DataFrame(L).to_csv('L_matrix.csv', index=False)
    pd.DataFrame(R).to_csv('R_matrix.csv', index=False)
    pd.DataFrame(B).to_csv('B_matrix.csv', index=False)
    pd.DataFrame([u], columns=['u']).to_csv('u_value.csv', index=False)
    pd.DataFrame(W).to_csv('W_matrix.csv', index=False)

T_op = np.block([
        [np.identity(m)],
        [np.zeros((n-m, m))]
    ])


k = np.linspace(500, 1, 500)
error_tri = []
error_ricci = []
C = np.ones((1, n))
base_f = f(0, A, B, C)

A_hat, H = hessenberg(A, calc_q=True)  # A_hat is tridiagonal, H is the transformation matrix
b_hat = H.T @ B
A_hat_inv = np.linalg.inv(A_hat)  # Compute the full inverse of A_hat
Phi = H @ np.diag(-(A_hat_inv @ b_hat).flatten())


def calculate_truncation_error(input_A, b, input_k):
    N = input_A.shape[0]
    H_k = H[:, :input_k]
    A_k = A_hat[:input_k, :input_k]
    b_k = b_hat[:input_k]
    A_k_inv = np.linalg.inv(A_k)
    Phi_k = np.zeros((N, N))
    Phi_k[:, :input_k] = H_k @ np.diag(-(A_k_inv @ b_k).flatten())

    error_matrix = Phi - Phi_k
    error = np.sqrt(n) * np.linalg.norm(error_matrix, ord=np.inf)

    return error


for i in k:
    A_red = T_op.T @ A @ T_op
    B_red = T_op.T @ B
    C_red = C @ T_op
    current_f = f(0, A_red, B_red, C_red)
    new_error = np.linalg.norm(base_f - current_f)
    error_ricci.append(new_error)
    T_op = np.delete(T_op, 0, axis=1)
    new_tri_error = m ** 0.25 * np.sum(calculate_truncation_error(A, B, int(i)))
    error_tri. append(new_tri_error)
    """
    current_op = 10000000000
    current_T = T_op
    for j in range(int(i)):
        test_t = np.delete(T_op, j, axis=1)
        A_red = test_t.T @ A @ test_t
        B_red = test_t.T @ B
        C_red = C @ test_t
        current_f = f(0, A_red, B_red, C_red)
        new_error = np.linalg.norm(base_f - current_f)
        if new_error < current_op:
            current_op = new_error
            current_T = np.delete(T_op, j, axis=1)
    T_op = current_T
    error_ricci.append(current_op)
    """


# Plot regression curves
fig, ax = plt.subplots()
ax.plot(np.flip(k), np.flip(error_ricci), color='b', label='Ricci-Optimal')
ax.plot(np.flip(k), np.flip(error_tri), color='r', label='Tri-Optimal')
ax.set_xlabel('Dimension (k)')
ax.set_ylabel('Total Error')
ax.set_title('Error vs Dimension Truncation')
ax.legend()
ax.grid(True)

plt.show()
