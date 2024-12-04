from bi_network import BidirectionalNetwork
import networkx as nx
import numpy as np
import pandas as pd
from persistent.network_db import NetworkDB


class NetworkGenerator:
    @staticmethod
    def generate_network(n: int, d: int, lp: (int, int), rp: (int, int),
                         network_db: NetworkDB) -> BidirectionalNetwork:
        # Generate a random graph with an average degree of approximately 70
        p = d / (n - 1)  # Probability to achieve average degree of 70
        graph = nx.gnp_random_graph(n, p)
        while not nx.is_connected(graph):  # Ensure the graph is connected
            graph = nx.gnp_random_graph(n, p)

        # Create adjacency matrix L from the graph
        laplacian = nx.to_numpy_array(graph)
        laplacian = laplacian * np.random.uniform(lp[0], lp[1], (n, n))  # Assign random weights to edges
        laplacian = (laplacian + laplacian.T) / 2  # Symmetrize L
        np.fill_diagonal(laplacian, -np.sum(laplacian, axis=1))

        rrate = np.diag(np.random.normal(rp[0], rp[1], n))  # Increase mean and variance for larger values

        a = - rrate + laplacian

        b = np.random.rand(n, 1)

        c = np.random.rand(n)

        u = 1  # Increase mean and variance for larger value

        d = BidirectionalNetwork.calculate_distance(a)

        w = BidirectionalNetwork.calculate_wasserstein_distance(a, b)

        if network_db is not None:
            filename: str = network_db.add_entry(n=n, d=d, lp=lp, rp=rp)
            # Save persistent to HDF5 file
            with pd.HDFStore(f'{filename}.h5') as store:
                store['L'] = pd.DataFrame(laplacian)
                store['R'] = pd.DataFrame(rrate)
                store['B'] = pd.DataFrame(b)
                store['u'] = pd.DataFrame([u], columns=['u'])
                store['W'] = pd.DataFrame(w)

        return BidirectionalNetwork(a, b, c, d)
