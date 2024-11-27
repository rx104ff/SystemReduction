import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def generate_symmetric_full_rank(nn):
    """
    Generates a random symmetric full-rank matrix of size n x n.
    """
    # Step 1: Create a random matrix
    A = np.random.randn(nn, nn)

    # Step 2: Make it symmetric by averaging with its transpose
    symmetric_matrix = (A + A.T) / 2

    # Step 3: Ensure full rank by adding a small positive value to the diagonal
    epsilon = 1e-3
    symmetric_matrix += epsilon * np.eye(nn)

    return symmetric_matrix


# Function to generate a random positive diagonal matrix with larger values
def random_positive_diagonal_matrix(n, low=100, high=1000):
    # Generate a random vector of values in the range [low, high]
    random_values = np.random.uniform(low, high, n)
    # Create a diagonal matrix using the random values
    diagonal_matrix = np.diag(random_values)
    return diagonal_matrix


n = 100# Generate a random graph with a variable average degree to achieve more variance
p = np.random.uniform(0.04, 0.1)  # Random probability for varying degrees
G = nx.gnp_random_graph(n, p)
while not nx.is_connected(G):  # Ensure the graph is connected
    G = nx.gnp_random_graph(n, p)
    # Calculate and print degree variance of G
degrees = [val for (node, val) in G.degree()]
degree_variance = np.var(degrees)
print(f"Degree variance of G: {degree_variance}")
degree_mean = np.mean(degrees)
print(f"Degree mean of G: {degree_mean}")
    # Create adjacency matrix L from the graph
L = nx.to_numpy_array(G)
L = L * np.random.uniform(0, 100, (n, n))  # Assign random weights to edges with a wider range
L = (L + L.T) / 2  # Symmetrize L
np.fill_diagonal(L, -np.sum(L, axis=1))  # Make diagonal negative sum of the rest of the row to ensure Metzler matrix

#R = np.diag(np.random.normal(10, 5, n))  # Increase mean and variance for larger values
R = np.diag(np.random.normal(50, 20, n))
#print(R)
#R = random_positive_diagonal_matrix(n)
#print(R)

A = -R + L

B = random_array = np.random.rand(n, 1)

u = 10
x = -np.linalg.solve(A, B * u)

kk = 4
T = np.zeros((n, kk))
k = n
for i in range(kk):
    if i != kk - 1:
        te = random.randint(1, k - (kk - i + 1))
    else:
        te = k
    for j in range(n-k, n-k + te):
        T[j, i] = 1/np.sqrt(te)
    k = k - te

T_ones = T
T_ones[T_ones != 0] = 1

c = np.ones(n)

c_s = c @ T @ np.linalg.inv(T.transpose() @ A @ T) @ T.transpose() @ A
c_s_2 = c @ T_ones @ np.linalg.inv(T_ones.transpose() @ A @ T_ones) @ T_ones.transpose() @ A

E = T @ np.linalg.inv(T.transpose() @ A @ T) @ T.transpose() @ A
M = T @ np.linalg.inv(T.T @ A @ T) @ T.T

x_1 = A@M
x_2 = M@A

tM = generate_symmetric_full_rank(n)

sM = np.random.randn(n, n)

print(M)
print(x)




#print(np.linalg.inv(A))
#print(np.linalg.inv(A)@np.ones((n, 1)))


#print(B * u)
# Display the graph G
#plt.figure(figsize=(10, 8))
#nx.draw(G, node_size=10, node_color='blue', edge_color='gray', with_labels=False)
#plt.title('Random Graph G')
#plt.show()
