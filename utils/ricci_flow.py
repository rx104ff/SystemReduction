import numpy as np


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
