import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

n = 1000

# Load L, R, B, u, W if they exist, otherwise calculate them
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
    np.fill_diagonal(L,
                     -np.sum(L, axis=1))  # Make diagonal negative sum of the rest of the row to ensure Metzler matrix

    R = np.diag(np.random.normal(10, 5, n))  # Increase mean and variance for larger values

    A = -R + L

    B = np.random.normal(10, 5, (n, 1))

    u = np.random.normal(10, 5)  # Increase mean and variance for larger value
    x = -np.linalg.solve(A, B * u)

    # Precompute (x[v] - x[u]) ** 2 for all pairs (u, v)
    x_diff_squared = np.zeros((n, n))
    for u_idx in range(n):
        for v in range(n):
            x_diff_squared[u_idx, v] = (x[v] - x[u_idx]) ** 2

    # Compute W(i, j) using a more efficient approach
    W = np.zeros((n, n))
    non_zero_indices = [np.nonzero(A[i, :])[0] for i in
                        range(n)]
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


def cost_function(A, C, clusters, x):
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

mean_D = np.mean(D[D != np.inf])
std_D = np.std(D[D != np.inf])

mean_A = np.mean(A)
std_A = np.std(A)
mean_B = np.mean(B)
std_B = np.std(B)
mean_C = np.mean(C)
std_C = np.std(C)

scale_factor = (mean_A + mean_B + mean_C) / 3
scaled_std = (std_A + std_B + std_C) / 3

initial_criterion = max(mean_D + 5000 * scaled_std * scale_factor, 1e-2)  # Ensure criterion is greater than 0
C_criterion = initial_criterion
k_values = []
all_clusterings = []

for target_k in [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 300, 400]:
    while True:
        print(f"Processing criterion = {C_criterion} for target k >= {target_k}...")
        clusters = [-1] * n
        current_cluster = 0

        for i in range(n):
            if clusters[i] == -1:  # If the node is not yet clustered
                clusters[i] = current_cluster
                for j in range(i + 1, n):
                    d = D[i, j]
                    if d < C_criterion:
                        clusters[j] = current_cluster
                current_cluster += 1

        k = current_cluster
        if k >= target_k:
            k_values.append(k)
            all_clusterings.append(clusters)
            best_Q = cost_function(A, C, clusters, x)
            print(f"Best Q for criterion = {C_criterion}: {best_Q}, resulting in k = {k}")
            break
        else:
            # Reduce the criterion to increase the number of clusters
            C_criterion = max(C_criterion / 1.15, 1e-8)  # Reduce to 1/10, ensuring criterion is greater than 0

s = 0

def f(s, A, B, C):
    epsilon = 1e-5  # Small regularization term to avoid singular matrix error
    try:
        return C @ np.linalg.inv(s * np.eye(A.shape[0]) - A + epsilon * np.eye(A.shape[0])) @ B
    except np.linalg.LinAlgError:
        # Handle singular matrix by adding a larger regularization term
        epsilon = 1e-3
        return C @ np.linalg.inv(s * np.eye(A.shape[0]) - A + epsilon * np.eye(A.shape[0])) @ B


# Compute A^t
t = 5.0  # Example value for t
A_t = compute_A_t(A, C, t)

# Clustering for A^t
x_t = -np.linalg.solve(A_t, B * u)

D_t = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        D_t[i, j] = np.abs(x_t[i] - x_t[j]) / A_t[i, j] if A_t[i, j] != 0 else np.inf
        D_t[j, i] = D_t[i, j]

C_criterion = initial_criterion
k_values_t = []
all_clusterings_t = []

for target_k in [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 300, 400, 500, 600]:
    while True:
        print(f"Processing criterion = {C_criterion} for target k >= {target_k} (A^t)...")
        clusters = [-1] * n
        current_cluster = 0

        for i in range(n):
            if clusters[i] == -1:  # If the node is not yet clustered
                clusters[i] = current_cluster
                for j in range(i + 1, n):
                    d = D_t[i, j]
                    if d < C_criterion:
                        clusters[j] = current_cluster
                current_cluster += 1

        k = current_cluster
        if k >= target_k:
            k_values_t.append(k)
            all_clusterings_t.append(clusters)
            best_Q = cost_function(A_t, C, clusters, x_t)
            print(f"Best Q for criterion = {C_criterion}: {best_Q}, resulting in k = {k}")
            break
        else:
            # Reduce the criterion to increase the number of clusters
            C_criterion = max(C_criterion / 1.15, 1e-8)  # Reduce to 1/10, ensuring criterion is greater than 0

errors = []
errors_t = []
for k_idx, (k, clustering, clustering_t) in enumerate(zip(k_values, all_clusterings, all_clusterings_t)):
    print(f"Computing error for k = {k} ({k_idx + 1}/{len(k_values)})...")
    T = np.zeros((n, k))  # Corrected dimensions for T
    for i in range(k):
        for j in range(n):
            if clustering[j] == i:
                T[j, i] = 1

    A_red = T.T @ A @ T
    B_red = T.T @ B
    C_red = C @ T

    error = f(s, A, B, C) - f(s, A_red, B_red, C_red)
    errors.append(np.linalg.norm(error))
    print(f"Error for k = {k}: {np.linalg.norm(error)}")

    T_t = np.zeros((n, k))  # Corrected dimensions for T
    for i in range(k):
        for j in range(n):
            if clustering_t[j] == i:
                T_t[j, i] = 1

    A_red_t = T_t.T @ A_t @ T_t
    B_red_t = T_t.T @ B
    C_red_t = C @ T_t

    error_t = f(s, A_t, B, C) - f(s, A_red_t, B_red_t, C_red_t)
    errors_t.append(np.linalg.norm(error_t))
    print(f"Error for k = {k} (A^t): {np.linalg.norm(error_t)}")

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
#ax.plot(k_values, errors, marker='o', color='b', label='A')

ax.plot(k_fit, errors_t_fit, linestyle='--', color='r', label='Polynomial Regression A^t')
#ax.plot(k_values_t, errors_t, marker='x', color='r', label='A^t')

ax.set_xlabel('Dimension (k)')
ax.set_ylabel('Error')
ax.set_title('Error vs Dimension Reduction for A and A^t with Monotonic Polynomial Regression Curves')
ax.legend()
ax.grid(True)

plt.show()
