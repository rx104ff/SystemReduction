from utils.ricci_flow import RicciFlow
from bi_network import BidirectionalNetwork
import numpy as np
from numpy.typing import NDArray
from itertools import combinations


class KOptimal:
    def __init__(self, network: BidirectionalNetwork):
        self.network = network

    @staticmethod
    def build_truncation(n, indices: NDArray[np.int64]) -> NDArray[np.int64]:
        trunc = np.zeros((n, len(indices)))
        for k, index in enumerate(indices):
            trunc[index, k] = 1
        return trunc

    @staticmethod
    def is_k_optimal(a, k_values):
        # Iterate over each row in A
        for row_index, row in enumerate(a):
            if row_index in k_values:  # Check if the row index is in the node array
                # Values with column indices in `nodes`
                in_nodes_values = row[k_values]
                min_in_nodes = np.min(in_nodes_values)

                # Values with column indices NOT in `nodes`
                complement_columns = np.setdiff1d(np.arange(a.shape[1]), k_values)  # Complement indices
                complement_values = row[complement_columns]
                max_complement = np.max(complement_values)

                if min_in_nodes < max_complement:
                    return False
        return True

    @staticmethod
    def l_one_norm(a, b):
        x = np.linalg.inv(-a, b)
        return np.sum(x)

    @staticmethod
    def l_two_norm(a, b):
        x = np.linalg.inv(-a, b)
        return np.sqrt(np.sum(x ** 2))

    def produce_max_optimal_at_t(self, t):
        forwarded_network = RicciFlow.compute_a_t(self.network.a, self.network.w, t)

        # Flatten the matrix to simplify finding top values
        flattened = forwarded_network.flatten()

        # Get the sorted indices
        sorted_indices = np.argsort(flattened)[::-1]  # Sort in descending order

        for k in range(self.network.n):
            if self.is_k_optimal(forwarded_network, sorted_indices[0: self.network.n - k]):
                return k, sorted_indices[0: self.network.n - k]

    def optimal_algorithm(self) -> (int, NDArray[np.int64]):
        num_points = 100
        t_values = np.linspace(0, 10, num_points)  # Range of t
        current_k = self.network.n
        ret = np.array([])
        for t in t_values:
            k, indices = self.produce_max_optimal_at_t(t)
            if current_k < k:
                ret = indices
            else:
                return k, ret

    def brutal_force(self, k):
        # Function to compute truncation matrix T
        def compute_truncation_matrix(index, n):
            T = np.zeros((n, len(index)))
            for idx, col in enumerate(index):
                T[col, idx] = 1
            return T

        # Brute force search for optimal T
        min_cost = float("inf")
        best_T = None
        best_indices = None

        for indices in combinations(range(self.network.n), k):
            T = compute_truncation_matrix(indices, self.network.n)
            a_truncated = T.T @ self.network.a @ T
            b_truncated = T.T @ self.network.b
            try:
                cost = np.linalg.norm(np.linalg.inv(-self.network.a) @ self.network.b -
                                      T @ np.linalg.inv(-a_truncated) @ b_truncated)
            except np.linalg.LinAlgError:
                cost = float("inf")  # Skip singular matrices

            if cost < min_cost:
                min_cost = cost
                best_T = T
                best_indices = indices

        return best_T, best_indices

    def close_region(self):
        pass

    def far_region(self):
        pass

    def complete_network(self):
        pass
