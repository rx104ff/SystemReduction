import numpy as np


class PositiveTridiagonalRealization:
    def __init__(self):
        pass

    @staticmethod
    def householder_by_vec(input_x, input_y):
        assert (len(input_x) == len(input_y))
        if len(input_x) == 1:
            return np.eye(1)
        dim = input_x.shape[0]
        ret = np.eye(dim) - 2 * np.outer(input_x - input_y, input_x - input_y) / (
                    np.linalg.norm(input_x - input_y) ** 2)
        return ret

    @staticmethod
    def tridiagonalization_k(a, b, k):
        dim = len(b)
        assert (k > 1)
        assert (k <= dim)

        b_tilde_1 = np.zeros(dim)
        b_tilde_1[0] = np.linalg.norm(b)

        h_0 = PositiveTridiagonalRealization.householder_by_vec(b, b_tilde_1)
        a_1 = h_0 @ a @ h_0

        a_k = a_1
        b_k = b_tilde_1
        ret_h = h_0

        for i in range(1, k):
            x = a_k[i - 1, i::]
            x_tilde = np.zeros(len(x))
            x_tilde[0] = np.linalg.norm(x)
            h_k_tilde = PositiveTridiagonalRealization.householder_by_vec(x, x_tilde)

            eye_k = np.eye(i)
            h_k = np.block([
                [eye_k, np.zeros((eye_k.shape[0], h_k_tilde.shape[1]))],
                [np.zeros((h_k_tilde.shape[0], eye_k.shape[1])), h_k_tilde]
            ])
            a_k = h_k @ a_k @ h_k
            b_k = h_k @ b_k
            ret_h = ret_h @ h_k

        return a_k, b_k, ret_h

    @staticmethod
    def tridiagonalization(a, b):
        return PositiveTridiagonalRealization.tridiagonalization_k(a, b, len(b))

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
        A_hat, b_hat, _ = PositiveTridiagonalRealization.tridiagonalization(A, b)

        print("Tridiagonal Matrix A_hat:")
        print(A_hat)
        print("\nTransformed Vector b_hat:")
        print(b_hat)

# To use the package:
# if __name__ == "__main__":
#     PositiveTridiagonalRealization.example_usage()
