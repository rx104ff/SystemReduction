import cvxpy as cp
import numpy as np
import random

import networkx as nx


# Function to generate a random positive diagonal matrix with larger values
def random_positive_diagonal_matrix(n, low=100, high=1000):
    # Generate a random vector of values in the range [low, high]
    random_values = np.random.uniform(low, high, n)
    # Create a diagonal matrix using the random values
    diagonal_matrix = np.diag(random_values)
    return diagonal_matrix


n = 20# Generate a random graph with a variable average degree to achieve more variance
p = np.random.uniform(0.04, 0.1)  # Random probability for varying degrees
G = nx.gnp_random_graph(n, p)
while not nx.is_connected(G):  # Ensure the graph is connected
    G = nx.gnp_random_graph(n, p)
    # Calculate and print degree variance of G

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

k = 10
"""
# Calculate the target value (sum of x)
target_sum = np.sum(x)

# Define variables
c = cp.Variable(n)
t = cp.Variable()

# Objective function to minimize |c^T x - 1_n^T x|
objective = cp.Minimize(t)

# Constraints
constraints = [
    t >= c @ x - target_sum,   # t >= c^T x - 1_n^T x
    t >= -(c @ x - target_sum), # t >= -(c^T x - 1_n^T x)
    cp.sum(c) == n,            # Sum constraint on c
    c >= 0,                    # Non-negativity constraint on c
    c <= n / (k-1)                 # Upper bound constraint on each c_i
]

# Apply the additional lower bound constraint to prevent trivial solution
for i in range(k):
    pass
    #constraints.append(c[i] >= n / (k))

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Display the results
if problem.status == cp.OPTIMAL:
    print("Minimum value of |c^T x - 1_n^T x|:", problem.value)
    print("Optimal vector c:", c.value)
else:
    print("The problem could not be solved to optimality.")
    print("Problem status:", problem.status)
    
"""

# Calculate the target value (sum of x)
target_sum = np.sum(x)

# Define a small threshold delta to avoid trivial solution
delta = 1.7 # Adjust this value based on your tolerance for diversity

# Define variables
c = cp.Variable(n)
t = cp.Variable()

# Objective function to minimize |c^T x - 1_n^T x|
objective = cp.Minimize(t)

# Constraints
constraints = [
    t >= c @ x - target_sum,    # t >= c^T x - 1_n^T x
    t >= -(c @ x - target_sum), # t >= -(c^T x - 1_n^T x)
    cp.sum(c) == n,             # Sum constraint on c
    c >= 0,                     # Non-negativity constraint on c
    c <= n / k                  # Upper bound constraint on each c_i
]

# Enforce the minimum value on a subset of entries to avoid trivial solutions
# Select k - 1 entries of c to have at least the value delta
for i in range(k - 1):
    constraints.append(c[i] >= delta)

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Display the results
if problem.status == cp.OPTIMAL:
    print("Minimum value of |c^T x - 1_n^T x| with subset minimum constraint:", problem.value)
    print("Optimal vector c:", c.value)
else:
    print("The problem could not be solved to optimality.")
    print("Problem status:", problem.status)
