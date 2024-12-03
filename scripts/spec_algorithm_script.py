import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def calculate_variance(D):
    return np.var(D[D != 0])


def compute_ricci_a(input_A, Kappa, t):
    ret = input_A.copy()
    for i in range(input_A.shape[0]):
        for j in range(input_A.shape[1]):
            if i != j:
                ret[i, j] = (input_A[i, j] * np.exp(t)) / (1 + Kappa[i, j] * (np.exp(t) - 1))
    return ret


def cost_function(A, clusters, x):
    def g(d):
        return 1 / d if d != 0 else 0

    num, denom = 0, 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            d = np.abs(x[i] - x[j]) / A[i, j] if A[i, j] != 0 else 0
            num += g(d) * (1 if clusters[i] == clusters[j] else 0)
            denom += g(d)
    return num / denom if denom != 0 else 0

def f(s, A, B, C):
    epsilon = 1e-5  # Small regularization term to avoid singular matrix error
    try:
        return C @ np.linalg.inv(s * np.eye(A.shape[0]) - A + epsilon * np.eye(A.shape[0])) @ B
    except np.linalg.LinAlgError:
        # Handle singular matrix by adding a larger regularization term
        epsilon = 1e-3
        return C @ np.linalg.inv(s * np.eye(A.shape[0]) - A + epsilon * np.eye(A.shape[0])) @ B

n = 1000

# Load L, R, B, u, W if they exist, otherwise calculate them
if os.path.exists('L_matrix.csv') and os.path.exists('R_matrix.csv') and os.path.exists('B_matrix.csv') and os.path.exists('u_value.csv') and os.path.exists('W_matrix.csv'):
    L = pd.read_csv('../L_matrix.csv').values
    R = pd.read_csv('../R_matrix.csv').values
    B = pd.read_csv('../B_matrix.csv').values
    #u = pd.read_csv('u_value.csv').values[0, 0]
    u = 20
    W = pd.read_csv('../W_matrix.csv').values
    A = -R + L
    x = -np.linalg.solve(A, B * u)
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
    np.fill_diagonal(L, -np.sum(L, axis=1))  # Make diagonal negative sum of the rest of the row to ensure Metzler matrix

    R = np.diag(np.random.normal(10, 5, n))  # Increase mean and variance for larger values

    A = -R + L

    B = random_array = np.random.rand(n, 1)

    u = 20  # Increase mean and variance for larger value
    x = -np.linalg.solve(A, B * u)

    # Precompute (x[v] - x[u]) ** 2 for all pairs (u, v)
    x_diff_squared = np.zeros((n, n))
    for u_idx in range(n):
        for v in range(n):
            x_diff_squared[u_idx, v] = (x[v] - x[u_idx]) ** 2

    # Compute W(i, j) using a more efficient approach
    W = np.zeros((n, n))
    non_zero_indices = [np.nonzero(A[i, :])[0] for i in range(n)]  # Precompute non-zero neighbors for each node

    for i in range(n):
        for j in range(n):
            if i != j:
                neighbors_i = non_zero_indices[i]
                neighbors_j = non_zero_indices[j]
                sum_neighbors = 0
                for u in neighbors_i:
                    for v in neighbors_j:
                        if A[u, v] != 0:
                            sum_neighbors += (x_diff_squared[u, v] / x[u]) + (x_diff_squared[v, j] / x[j])
                W[i, j] = sum_neighbors

    # Save L, R, B, u, W to CSV files
    pd.DataFrame(L).to_csv('L_matrix.csv', index=False)
    pd.DataFrame(R).to_csv('R_matrix.csv', index=False)
    pd.DataFrame(B).to_csv('B_matrix.csv', index=False)
    pd.DataFrame([u], columns=['u']).to_csv('u_value.csv', index=False)
    pd.DataFrame(W).to_csv('W_matrix.csv', index=False)

# Compute D matrix as D_ij = |x_j - x_i| / A_ij
D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if A[i, j] != 0:
            D[i, j] = np.abs(x[j] - x[i]) / A[i, j]
        else:
            D[i, j] = 0

K = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i != j:
            K[i, j] = W[i, j] / D[i, j]

A_t = compute_ricci_a(A, K, 2.0)
D_t = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if A_t[i, j] != 0:
            D_t[i, j] = np.abs(x[j] - x[i]) / A_t[i, j]
        else:
            D_t[i, j] = 0

# Test various values of k and compute the corresponding errors
k_values = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
errors = []
errors_t = []
C = random_array = np.random.rand(1, n)

print(calculate_variance(D))
for k in k_values:
    spectral = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=42)
    clusters = spectral.fit_predict(D)
    clusters_t = spectral.fit_predict(D_t)

    # Create the reduction matrix T based on the clustering
    T = np.zeros((n, k))
    T_t = np.zeros((n, k))
    for i in range(k):
        for j in range(n):
            if clusters[j] == i:
                T[j, i] = 1
            if clusters_t[j] == i:
                T_t[j, i] = 1

    # Calculate the reduced matrices
    A_red = T.T @ A @ T
    B_red = T.T @ B
    C_red = C @ T

    # Calculate the reduced matrices
    A_red_t = T_t.T @ A @ T_t
    B_red_t = T_t.T @ B
    C_red_t = C @ T_t

    # Compute the error using function f
    original_f = f(0, A, B, C)
    reduced_f = f(0, A_red, B_red, C_red)
    error = np.linalg.norm(original_f - reduced_f)
    errors.append(error)
    print(f"Error for spectral clustering with k = {k}: {error}")

    reduced_f_t = f(0, A_red_t, B_red_t, C_red_t)
    error_t = np.linalg.norm(original_f - reduced_f_t)
    errors_t.append(error_t)
    print(f"Error for spectral clustering with k = {k}: {error_t}")

# Plot the error vs k
plt.figure(figsize=(10, 6))
plt.plot(k_values, errors, marker='o', linestyle='-', color='b')
plt.plot(k_values, errors_t, marker='x', linestyle='dotted', color='g')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Error')
plt.title('Error vs Number of Clusters for Spectral Clustering')
plt.grid(True)
plt.show()
