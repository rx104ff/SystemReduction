import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.sparse.csgraph import connected_components

n = 1000


def find_off_diagonal_negatives(matrix):
    # Get the off-diagonal indices
    off_diag_indices = np.where(~np.eye(matrix.shape[0], dtype=bool))

    # Extract the off-diagonal elements
    off_diag_elements = matrix[off_diag_indices]

    # Filter the negative values
    negative_values = off_diag_elements[off_diag_elements < 0]

    return negative_values

# Load L, R, B, u, W if they exist, otherwise calculate them
if os.path.exists('L_matrix.csv') and os.path.exists('R_matrix.csv') and os.path.exists('B_matrix.csv') and os.path.exists('u_value.csv') and os.path.exists('W_matrix.csv'):
    L = pd.read_csv('../L_matrix.csv').values
    R = pd.read_csv('../R_matrix.csv').values
    B = pd.read_csv('../B_matrix.csv').values
    u = pd.read_csv('../u_value.csv').values[0, 0]
    W = pd.read_csv('../W_matrix.csv').values
    A = -R + L
else:
    # Generate a random graph with an average degree of approximately 70
    p = 70 / (n - 1)  # Probability to achieve average degree of 70
    G = nx.gnp_random_graph(n, p)
    while not nx.is_connected(G):  # Ensure the graph is connected
        G = nx.gnp_random_graph(n, p)

    # Create adjacency matrix L from the graph
    L = nx.to_numpy_array(G)
    L = L * np.random.uniform(0, 0.1, (n, n))  # Assign random weights to edges
    L = (L + L.T) / 2  # Symmetrize L
    np.fill_diagonal(L, -np.sum(L, axis=1))  # Make diagonal negative sum of the rest of the row to ensure Metzler matrix

    R = np.diag(np.random.normal(10, 5, n))
    R = np.abs(R)

    A = -R + L

    B = np.random.normal(10, 5, (n, 1))
    B = np.abs(B)

    u = 10  # Increase mean and variance for larger value
    x = -np.linalg.solve(A, B * u)

    if (x < 0).sum() > 0:
        print((x < 0).sum())
        exit(1)

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


# Compute C(i, j) using W(i, j) and |x_j - x_i|
x = -np.linalg.solve(A, B * u)
C = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            C[i, j] = W[i, j] / np.abs(x[j] - x[i]) if np.abs(x[j] - x[i]) != 0 else 0

P = A * C


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


def compute_A_t(A, C, t):
    A_t = A.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i != j:
                A_t[i, j] = (A[i, j] * np.exp(t)) / (1 + A[i, j] * C[i, j] * (np.exp(t) - 1))
    return A_t


L_u = np.zeros_like(L)
L_u[L != 0] = 1  # Set off-diagonal to 1 where L_ij != 0
np.fill_diagonal(L_u, -np.sum(L_u, axis=1))  # Set diagonal to -sum of the off-diagonal

D = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        D[i, j] = np.abs(x[i] - x[j]) / A[i, j] if A[i, j] != 0 else np.inf
        D[j, i] = D[i, j]

K = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i != j:
            K[i, j] = W[i, j] / D[i, j]

#PP = (np.log((A*C - 1)/(A*C)))
#PP = PP[np.isfinite(PP) & (PP < 0)]


t = 1  # Example value for t

A_t = compute_A_t(A, K, t)
D_t = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        if i != j:
            D_t[i, j] = np.abs(x[i] - x[j]) / A_t[i, j] if A_t[i, j] != 0 else np.inf
            D_t[j, i] = D_t[i, j]


ss = find_off_diagonal_negatives(A_t)

mean_D = np.mean(D[D != np.inf])
std_D = np.std(D[D != np.inf])
mean_D_t = np.mean(D[D != np.inf])
std_D_t = np.std(D[D != np.inf])

mean_A = np.mean(A)
std_A = np.std(A)
mean_B = np.mean(B)
std_B = np.std(B)
mean_C = np.mean(C)
std_C = np.std(C)

min_D = np.min(D[D != 0])
max_D = np.max(D[D != np.inf])
min_D_t = np.min(D_t[D_t != 0])
max_D_t = np.max(D_t[D_t != np.inf])

scale_factor = (mean_A + mean_B + mean_C) / 3
scaled_std = (std_A + std_B + std_C) / 3

initial_criterion = min_D  # Ensure criterion is greater than 0
initial_criterion_t = min_D_t
C_criterion = initial_criterion
k_values_base = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 110, 120, 130, 140, 150, 200]
k_values_base.reverse()
k_values = []
all_clusterings = []

for target_k in k_values_base:
    while True:
        print(f"Processing criterion = {C_criterion} for target k <= {target_k}...")
        # Set edges in D greater than criterion to infinity and update A accordingly
        D_cut = D.copy()
        D_cut[D_cut > C_criterion] = np.inf

        A_cut = A.copy()
        for i in range(n):
            for j in range(i + 1, n):
                if D_cut[i, j] == np.inf:
                    A_cut[i, j] = 0
                    A_cut[j, i] = 0

        # Compute the Laplacian matrix
        L_cut = np.diag(np.sum(A_cut, axis=1)) - A_cut

        # Calculate the number of connected components using scipy's connected_components
        _, labels = connected_components(csgraph=A_cut, directed=False, connection='weak')
        k = len(np.unique(labels))

        if k <= target_k:
            k_values.append(k)
            all_clusterings.append(labels)
            print(f"Criterion = {C_criterion}: resulting in k = {k}")
            break
        else:
            # Reduce the criterion to increase the number of clusters
            C_criterion = min(C_criterion * 1.15, max_D)  # Reduce to 1/10, ensuring criterion is greater than 0

# Calculate clustering matrix from the labels
C_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if labels[i] == labels[j]:
            C_matrix[i, j] = 1

# Create reduced A from the clustering matrix
A_reduced = C_matrix.T @ A @ C_matrix

s = 0


def f(s, A, B, C):
    epsilon = 1e-5  # Small regularization term to avoid singular matrix error
    try:
        return C @ np.linalg.inv(s * np.eye(A.shape[0]) - A + epsilon * np.eye(A.shape[0])) @ B
    except np.linalg.LinAlgError:
        # Handle singular matrix by adding a larger regularization term
        epsilon = 1e-3
        return C @ np.linalg.inv(s * np.eye(A.shape[0]) - A + epsilon * np.eye(A.shape[0])) @ B


# Clustering for A^t
x_t = -np.linalg.solve(A_t + 1e-5 * np.eye(A_t.shape[0]), B * u)  # Regularize to avoid singular matrix

D_t = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        D_t[i, j] = np.abs(x_t[i] - x_t[j]) / A_t[i, j] if A_t[i, j] != 0 else np.inf
        D_t[j, i] = D_t[i, j]

C_criterion = initial_criterion_t
k_values_t = []
all_clusterings_t = []

for target_k in k_values_base:
    while True:
        print(f"Processing criterion = {C_criterion} for target k <= {target_k} (A^t)...")
        # Set edges in D_t greater than criterion to infinity and update A_t accordingly
        D_t_cut = D_t.copy()
        D_t_cut[D_t_cut > C_criterion] = np.inf

        A_t_cut = A_t.copy()
        for i in range(n):
            for j in range(i + 1, n):
                if D_t_cut[i, j] == np.inf:
                    A_t_cut[i, j] = 0
                    A_t_cut[j, i] = 0

        # Compute the Laplacian matrix
        L_t_cut = np.diag(np.sum(A_t_cut, axis=1)) - A_t_cut

        # Calculate the number of connected components using scipy's connected_components
        _, labels_t = connected_components(csgraph=A_t_cut, directed=False, connection='weak')
        k = len(np.unique(labels_t))

        if k <= target_k:
            k_values_t.append(k)
            all_clusterings_t.append(labels_t)
            print(f"Criterion = {C_criterion}: resulting in k = {k}")
            break
        else:
            # Reduce the criterion to increase the number of clusters
            C_criterion = min(C_criterion * 1.15, max_D_t)   # Reduce to 1/10, ensuring criterion is greater than 0


errors = []
errors_t = []
c = random_array = np.random.rand(1, n)
for k_idx, (k, clustering, clustering_t) in enumerate(zip(k_values, all_clusterings, all_clusterings_t)):
    print(f"Computing error for k = {k} ({k_idx + 1}/{len(k_values)})...")
    T = np.zeros((n, k))  # Corrected dimensions for T
    for i in range(k):
        for j in range(n):
            if clustering[j] == i:
                T[j, i] = 1

    A_red = T.T @ A @ T
    B_red = T.T @ B
    c_red = c @ T

    C_t = T.T @ C @ T

    error = np.linalg.norm(f(s, A, B, c) - f(s, A_red, B_red, c_red))
    errors.append(error)
    print(f"Error for k = {k}: {error}")

    T_t = np.zeros((n, k))  # Corrected dimensions for T
    for i in range(k):
        for j in range(n):
            if clustering_t[j] == i:
                T_t[j, i] = 1

    A_red_t = T_t.T @ A @ T_t
    B_red_t = T_t.T @ B
    c_red_t = c @ T_t

    #A_rewinded = rewind_A_t(A_red_t, C_t, t)

    error_t = np.linalg.norm(f(s, A, B, c) - f(s, A_red_t, B_red_t, c_red_t))
    errors_t.append(error_t)
    print(f"Error for k = {k} (A^t): {error_t}")

print(f"The variance for A is {np.var(A)}")
print(f"The variance for A^t is {np.var(A_t)}")
print(f"The mean for A is {np.var(A)}")
print(f"The mean for A^t is {np.var(A_t)}")
# Ensure the lengths of k_values and k_values_t match for fitting
min_length = min(len(k_values), len(k_values_t))
k_values = k_values[:min_length]
k_values_t = k_values_t[:min_length]
errors = errors[:min_length]
errors_t = errors_t[:min_length]

# Fit polynomial regression using sklearn
poly = PolynomialFeatures(degree=3)

# Fit for errors of A
k_values_reshaped = np.array(k_values).reshape(-1, 1)
errors_reshaped = np.array(errors).reshape(-1, 1)
k_poly = poly.fit_transform(k_values_reshaped)
model = LinearRegression()
model.fit(k_poly, errors_reshaped)
k_fit = np.linspace(min(k_values), max(k_values), 500).reshape(-1, 1)
k_fit_poly = poly.transform(k_fit)
errors_fit = model.predict(k_fit_poly)

# Cap the predicted values within the range of actual errors to prevent overfitting
errors_fit = np.clip(errors_fit, a_min=np.min(errors), a_max=np.max(errors))

# Ensure monotonicity by enforcing a decreasing constraint
for i in range(1, len(errors_fit)):
    if errors_fit[i] > errors_fit[i - 1]:
        errors_fit[i] = errors_fit[i - 1]

# Fit for errors of A^t
errors_t_reshaped = np.array(errors_t).reshape(-1, 1)
model_t = LinearRegression()
k_poly_t = poly.fit_transform(k_values_reshaped)
model_t.fit(k_poly_t, errors_t_reshaped)
errors_t_fit = model_t.predict(k_fit_poly)

# Cap the predicted values within the range of actual errors_t to prevent overfitting
errors_t_fit = np.clip(errors_t_fit, a_min=np.min(errors_t), a_max=np.max(errors_t))

# Ensure monotonicity by enforcing a decreasing constraint
for i in range(1, len(errors_t_fit)):
    if errors_t_fit[i] > errors_t_fit[i - 1]:
        errors_t_fit[i] = errors_t_fit[i - 1]

# Plot regression curves
fig, ax = plt.subplots()

ax.plot(k_fit, errors_fit, linestyle='--', color='b', label='Polynomial Regression A')
ax.plot(k_values, errors, marker='o', color='b', label='A')

ax.plot(k_fit, errors_t_fit, linestyle='--', color='r', label='Polynomial Regression A^t')
ax.plot(k_values, errors_t, marker='x', color='r', label='A^t')

ax.set_xlabel('Dimension (k)')
ax.set_ylabel('Error')
ax.set_title('Error vs Dimension Reduction for A and A^t with Monotonic Polynomial Regression Curves')
ax.legend()
ax.grid(True)

plt.show()
