import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
from sklearn.cluster import SpectralClustering
n = 5

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
L = -L
R = np.diag(np.random.normal(10, 5, n))  # Increase mean and variance for larger values

A = -(20*R + L)
b = np.ones(n)
x = np.linalg.inv(-A) @ b

print(x)
print(-A)
