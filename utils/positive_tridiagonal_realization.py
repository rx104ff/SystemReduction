import numpy as np


class PositiveTridiagonalRealization:
    def __init__(self):
        pass

    @staticmethod
    def householder_transformation(b):
        """
        Constructs a Householder transformation matrix H such that H^T b results
        in a vector with only the first entry non-zero and positive.

        Parameters:
            b (ndarray): Input vector.

        Returns:
            H (ndarray): Orthogonal transformation matrix.
        """
        n = len(b)
        e1 = np.zeros_like(b)
        e1[0] = np.linalg.norm(b)
        v = b - e1
        v /= np.linalg.norm(v)  # Normalize v

        H = np.eye(n) - 2 * np.outer(v, v)  # Householder matrix
        return H

    @staticmethod
    def positive_tridiagonal_realization(A, b):
        """
        Compute a positive tridiagonal realization of the pair (A, b).

        Parameters:
            A (ndarray): Symmetric positive definite matrix of size (n, n).
            b (ndarray): Vector of size (n,) corresponding to the network.

        Returns:
            A_hat (ndarray): Tridiagonal matrix with positive subdiagonal entries.
            b_hat (ndarray): Transformed vector with a single non-zero positive entry at the beginning.
        """
        n = A.shape[0]
        assert A.shape == (n, n), "Matrix A must be square."
        assert np.allclose(A, A.T), "Matrix A must be symmetric."
        assert b.shape == (n,), "Vector b must have the same length as A's dimensions."

        # Construct the Householder matrix H
        H = PositiveTridiagonalRealization.householder_transformation(b)

        # Transform A to tridiagonal form
        A_hat = H.T @ A @ H

        # Adjust beta values to ensure positivity
        alpha = np.diag(A_hat)
        beta = np.diag(A_hat, k=1)
        beta = np.abs(beta)  # Ensure positive subdiagonal entries

        # Reconstruct the tridiagonal A_hat with adjusted beta
        A_hat_tridiagonal = np.zeros_like(A_hat)
        np.fill_diagonal(A_hat_tridiagonal, alpha)
        np.fill_diagonal(A_hat_tridiagonal[1:], beta)
        np.fill_diagonal(A_hat_tridiagonal[:, 1:], beta)

        # Transform b to b_hat
        b_hat = H.T @ b

        # Ensure b_hat has only the first entry non-zero and positive
        b_hat[1:] = 0
        b_hat[0] = np.linalg.norm(b)  # First entry must be positive

        return A_hat_tridiagonal, b_hat

    @staticmethod
    def example_usage():
        """ Example usage for demonstration. """
        # Define a symmetric positive definite matrix A
        A = np.array([[-4, 1, 0],
                      [1, -3, 1],
                      [0, 1, -2]])

        # Define vector b
        b = np.array([1, 2, 3])

        # Compute the positive tridiagonal realization
        A_hat, b_hat = PositiveTridiagonalRealization.positive_tridiagonal_realization(A, b)

        print("Tridiagonal Matrix A_hat:")
        print(A_hat)
        print("\nTransformed Vector b_hat:")
        print(b_hat)

# To use the package:
# if __name__ == "__main__":
#     PositiveTridiagonalRealization.example_usage()
