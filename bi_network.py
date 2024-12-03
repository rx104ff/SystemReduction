from typing import Self

import numpy as np
from numpy.typing import NDArray


class BidirectionalNetwork:
    """
    Instantiate a BidirectionalNetwork.
    Calculate distance and Wasserstein distance matrix if not given

    Parameters:
        a (ndarray): Network matrix.
        b (ndarray): Input vector.
        c (ndarray): Output vector.
        d (ndarray): Optional distance vector.
        w (ndarray): Optional Wasserstein distance vector.

    Returns:
        B (BidirectionalNetwork): Class instance
    """
    def __init__(self, a: np.array, c: np.array, b: np.array, d: np.array = None, w: np.array = None):
        assert(a.shape[0] == a.shape[1])
        assert(a.shape[1] == b.shape[0])
        assert(a.shape[0] == c.shape[0])
        if not np.allclose(a, a.T):
            raise ValueError("Input matrix must be symmetric.")
        self.a = a
        self.b = b
        self.c = c
        self.n = a.shape[0]

        if d is not None:
            self.d = d
        else:
            self.d = self.calculate_distance(a)

        if w is not None:
            self.w = w
        else:
            self.w = self.calculate_wasserstein_distance(a, b)

    def adj_matrix(self) -> NDArray[np.int64]:
        # Threshold to create adjacency matrix
        adjacency_matrix = (self.a > 0).astype(int)

        # Remove self-loops (diagonal elements)
        np.fill_diagonal(adjacency_matrix, 0)

        return adjacency_matrix

    def remove_node(self, k) -> Self:
        self.a = np.delete(np.delete(self.a, k, axis=0), k, axis=1)
        self.b = np.delete(self.b, k, axis=0)
        self.c = np.delete(self.c, k, axis=0)
        return self

    @staticmethod
    def calculate_wasserstein_distance(a, b):
        x = np.linalg.inv(-a, b)
        n = a.shape[0]

        # Compute W(i, j) using a more efficient approach
        w = np.zeros((n, n))
        non_zero_indices = [np.nonzero(a[i, :])[0] for i in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    neighbors_i = non_zero_indices[i]
                    neighbors_j = non_zero_indices[j]
                    sum_neighbors = 0
                    for u in neighbors_i:
                        for v in neighbors_j:
                            if a[u, v] != 0:
                                sum_neighbors += (1 / x[u]) + (1 / x[j])
                    w[i, j] = sum_neighbors

        return w

    @staticmethod
    def calculate_distance(a):
        n = a.shape[0]
        d = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    if a[i, j] != 0:
                        d[i, j] = 1 / a[i,j]

        return d