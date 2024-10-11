import numpy as np
import matplotlib.pyplot as plt

n = 2000
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

initial_criterion = max(mean_D + scaled_std * scale_factor, 1e-3)  # Ensure criterion is greater than 0
C_criterion = initial_criterion
k_values = []
all_clusterings = []

for target_k in [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 110, 120, 130, 140, 150, 200]:
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
    return C @ np.linalg.inv(s * np.eye(A.shape[0]) - A) @ B


errors = []
for k_idx, (k, clustering) in enumerate(zip(k_values, all_clusterings)):
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
