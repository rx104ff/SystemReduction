import numpy as np
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def f(s, A, B, C):
    return C @ np.linalg.inv(s * np.eye(A.shape[0]) - A) @ B


def spectral_clustering(graph, edge_weights, steady_states, g, k):
    V = len(graph)  # Number of nodes
    E = [(i, j) for i, neighbors in enumerate(graph) for j in neighbors if i < j]  # Edges (i, j), i < j

    # Step 1: Initialize distance matrix D with zeros
    D = np.zeros((V, V))

    # Step 2: Compute distance matrix D
    for i, j in E:
        d_ij = abs(steady_states[j] - steady_states[i]) / edge_weights[i, j]
        D[i, j] = d_ij
        D[j, i] = d_ij  # Since the graph is undirected

    # Step 3: Compute similarity matrix S
    S = np.zeros((V, V))
    for i, j in E:
        S_ij = g(D[i, j])
        S[i, j] = S_ij
        S[j, i] = S_ij

    # Step 4: Compute degree matrix W
    W = np.zeros((V, V))
    for i in range(V):
        W[i, i] = np.sum(S[i, :])

    # Step 5: Compute normalized Laplacian L_sym
    W_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(W)))
    L_sym = np.eye(V) - W_inv_sqrt @ S @ W_inv_sqrt

    # Step 6: Compute first k eigenvectors of L_sym
    eigenvalues, eigenvectors = spla.eigsh(L_sym, k=k, which='SM')

    # Step 7: Form matrix U and normalize rows
    U = eigenvectors
    U_normalized = U / np.linalg.norm(U, axis=1, keepdims=True)

    # Step 8: Apply k-means clustering to rows of U
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    C = kmeans.fit_predict(U_normalized)

    # Output: Cluster assignments C(i) for all nodes i
    return C


def compute_a_ij_t(A, C, t):
    V = A.shape[0]
    a_t = np.zeros((V, V))
    for i in range(V):
        for j in range(V):
            if i != j:
                a_t[i, j] = (A[i, j] * np.exp(t)) / (1 + A[i, j] * C[i, j] * (np.exp(t) - 1))
    return a_t


def weighted_to_unweighted_laplacian(L_weighted):
    V = L_weighted.shape[0]
    L_unweighted = np.zeros((V, V))
    for i in range(V):
        for j in range(V):
            if i != j and L_weighted[i, j] != 0:
                L_unweighted[i, j] = -1
        L_unweighted[i, i] = -np.sum(L_unweighted[i, :])
    return L_unweighted


# Example usage with the given initialization
n = 1000
L = np.random.uniform(0, 50, (n, n))  # Ensure non-negative off-diagonal elements for Metzler property
L = (L + L.T) / 2  # Symmetrize L
np.fill_diagonal(L, -np.sum(L, axis=1))  # Make diagonal negative sum of the rest of the row to ensure Metzler matrix

R = np.diag(np.random.normal(10, 5, n))  # Increase mean and variance for larger values

A = -R + L

B = np.random.normal(10, 5, (n, 1))
C = np.random.normal(10, 5, (1, n))


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


u = np.random.normal(10, 5)  # Increase mean and variance for larger value
x = -np.linalg.solve(A, B * u)

# Perform spectral clustering on the graph defined by A
k_values = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 110, 120, 130, 140, 150, 200]
all_clusterings = []
errors = []

for k in k_values:
    print(f"Performing spectral clustering with k = {k}...")
    clusters = spectral_clustering(graph=[[j for j in range(n) if A[i, j] != 0] for i in range(n)],
                                   edge_weights=A, steady_states=x, g=lambda d: np.exp(-d), k=k)
    all_clusterings.append(clusters)

    # Dimension reduction
    T = np.zeros((n, k))
    for j in range(n):
        T[j, clusters[j]] = 1

    # Normalize columns of T
    for i in range(k):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) > 0:
            T[cluster_indices, i] = 1 / np.sqrt(len(cluster_indices))

    # Use pseudo-inverse for reduction
    T_pinv = np.linalg.pinv(T)
    A_red = T_pinv @ A @ T
    B_red = T_pinv @ B
    C_red = C @ T

    error = f(0, A, B, C) - f(0, A_red, B_red, C_red)
    errors.append(np.linalg.norm(error))
    print(f"Error for k = {k}: {np.linalg.norm(error)}")

fig, ax1 = plt.subplots()

ax1.plot(k_values, errors, marker='o', color='b')
ax1.set_xlabel('Dimension (k)')
ax1.set_ylabel('Error', color='b')
ax1.set_title('Error vs Dimension Reduction')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(k_values, errors, marker='o', color='r')
ax2.set_yscale('log')
ax2.set_ylabel('Error (log scale)', color='r')

plt.show()
