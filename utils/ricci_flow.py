import numpy as np
from tqdm import tqdm


class RicciFlow:
    @staticmethod
    # Function definitions
    def compute_a_t(a, w, t):
        """
        Compute a(i,j)^t = Aij * e^t / (1 + Aij * W(i,j) * (e^t - 1)).
        """
        numerator = a * np.exp(t)
        denominator = 1 + a * w * (np.exp(t) - 1)
        return numerator / denominator

    @staticmethod
    def compute_da_t(a, w, t):
        """
        Compute the derivative of a(i,j)^t with respect to t:
        da(i,j)^t/dt = |(Aij * e^t * (Aij * Wij - 1)) / (1 + Aij * Wij * (e^t - 1))^2|
        """
        numerator = a * np.exp(t) * (a * w - 1)
        denominator = (1 + a * w * (np.exp(t) - 1)) ** 2
        return np.abs(numerator / denominator)

    @staticmethod
    def compute_w(x):
        n = len(x)
        w = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    w[i, j] = 1 / x[i] + 1 / x[j]
        return w

    @staticmethod
    def update_beta(x, eta):
        l1_norm_x = np.sum(x)
        l_infty_norm_x = np.max(x)
        beta = eta * l1_norm_x * np.sum(
            [(l_infty_norm_x - x_j) * x_j / (l_infty_norm_x + x_j) for x_j in x]
        )
        return beta

    @staticmethod
    def update_a(a, x, beta, delta):
        w = RicciFlow.compute_w(x)
        eta = np.median(a * w)
        r = np.zeros(a.shape[0])
        diagonal_update = - beta / np.sum(x)
        for i in range(len(x)):
            initial_sum = 0
            for j in range(len(x)):
                if i != j:
                    initial_sum += a[i, j]
            r[i] = - (a[i, i] + initial_sum)

        for i in range(len(x)):
            off_diag_sum = 0
            for j in range(len(x)):
                if i != j:
                    a[i, j] = (eta * a[i, j] * np.exp(delta)) / (
                            eta + a[i, j] * w[i, j] * (np.exp(delta) - 1)
                    )
                    off_diag_sum += (eta * a[i, j] * np.exp(delta) * x[j]) / (
                            eta + a[i, j] * w[i, j] * (np.exp(delta) - 1)
                    )
            a[i, i] = a[i, i] * (1 - delta) + delta * (diagonal_update - (1 / x[i]) * off_diag_sum)
        return a

    @staticmethod
    def update_b(b, x, delta):
        l1_norm_b = np.sum(b)
        l1_norm_x = np.sum(x)

        normalized_b = b / l1_norm_b
        normalized_x = x / l1_norm_x

        update = b * np.exp(delta * (normalized_x - normalized_b))

        return update

    @staticmethod
    def flow(a, b, max_iters, delta, tolerance):
        for t in tqdm(range(max_iters), desc="Processing"):
            x_tilde = -np.linalg.solve(a, b)

            eta = np.median(a * RicciFlow.compute_w(x_tilde))
            beta = RicciFlow.update_beta(x_tilde, eta)

            b_new = RicciFlow.update_b(b, x_tilde, beta)
            a = RicciFlow.update_a(a, x_tilde, beta, delta)

            if np.linalg.norm(b - b_new) < tolerance:
                return a, b, t

            b = b_new

        return a, b, max_iters
