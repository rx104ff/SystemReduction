from bi_network import BidirectionalNetwork
import networkx as nx
import numpy as np
import pandas as pd


class NetworkGenerator:
    @staticmethod
    def generate_network(n: int, d: int, save: bool = True) -> BidirectionalNetwork:
        # Generate a random graph with an average degree of approximately 70
        p = d / (n - 1)  # Probability to achieve average degree of 70
        graph = nx.gnp_random_graph(n, p)
        while not nx.is_connected(graph):  # Ensure the graph is connected
            graph = nx.gnp_random_graph(n, p)

        # Create adjacency matrix L from the graph
        laplacian = nx.to_numpy_array(graph)
        laplacian = laplacian * np.random.uniform(0, 50, (n, n))  # Assign random weights to edges
        laplacian = (laplacian + laplacian.T) / 2  # Symmetrize L
        np.fill_diagonal(laplacian, -np.sum(laplacian, axis=1))  # Make diagonal negative sum of the rest of the row to ensure Metzler matrix

        rrate = np.diag(np.random.normal(10, 5, n))  # Increase mean and variance for larger values

        a = - rrate + laplacian

        b = np.random.rand(n, 1)

        c = np.random.rand(n)

        u = 20  # Increase mean and variance for larger value

        d = BidirectionalNetwork.calculate_distance(a)

        w = BidirectionalNetwork.calculate_wasserstein_distance(a, b)

        if save:
            # Save persistent to HDF5 file
            with pd.HDFStore('persistent.h5') as store:
                store['L'] = pd.DataFrame(laplacian)
                store['R'] = pd.DataFrame(rrate)
                store['B'] = pd.DataFrame(b)
                store['u'] = pd.DataFrame([u], columns=['u'])
                store['W'] = pd.DataFrame(w)

        return BidirectionalNetwork(a, b, c, d)
