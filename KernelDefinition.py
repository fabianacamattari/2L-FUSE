
"""
Classes and functions adopted with modifications from the folder freely available at 
https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/paper-2023-data-driven-kernel-designs.

Reference:
    Tizian Wenzel, Francesco Marchetti, and Emma Perracchione. 
    "Data-driven kernel designs for optimized greedy schemes: A machine learning perspective." 
    SIAM J. Sci. Comput., 46(1):C101–C126, 2024.
    https://doi.org/10.1137/23M1551201.

"""

import torch
from abc import ABC, abstractmethod
import numpy as np

# Abstract kernel
class Kernel(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self, x, y):
        pass

    def eval_prod(self, x, y, v, batch_size=100):
        N = x.shape[0]
        num_batches = int(np.ceil(N / batch_size))
        mat_vec_prod = np.zeros((N, 1))
        for idx in range(num_batches):
            idx_begin = idx * batch_size
            idx_end = (idx + 1) * batch_size
            A = self.eval(x[idx_begin:idx_end, :], y)
            mat_vec_prod[idx_begin:idx_end] = A @ v
        return mat_vec_prod

    @abstractmethod
    def diagonal(self, X):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass


# Abstract RBF
class RBF(Kernel):
    @abstractmethod
    def __init__(self):
        super(RBF, self).__init__()

    def eval(self, x, y):
      return self.rbf(self.ep,torch.cdist(x, y))

    def diagonal(self, X):
        return torch.ones(X.shape[0], 1) * self.rbf(self.ep, torch.tensor(0.0))

    def __str__(self):
        return self.name + ' [gamma = %2.2e]' % self.ep

    def set_params(self, par):
        self.ep = par

class Matern(RBF):
    def __init__(self, ep=1, k=0):
        self.ep = ep
        if k == 0:
            self.name = 'mat0'
            self.rbf = lambda ep, r: torch.exp(-ep * r)
        elif k == 1:
            self.name = 'mat1'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (1 + ep * r)
        elif k == 2:
            self.name = 'mat2'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (3 + 3 * ep * r + (ep * r) ** 2)
        elif k == 3:
            self.name = 'mat3'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (15 + 15 * ep * r + 6 * (ep * r) ** 2 + 1 * (ep * r) ** 3)
        elif k == 4:
            self.name = 'mat4'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (
                        105 + 105 * ep * r + 45 * (ep * r) ** 2 + 10 * (ep * r) ** 3 + 1 * (ep * r) ** 4)
        elif k == 5:
            self.name = 'mat5'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (
                        945 + 945 * ep * r + 420 * (ep * r) ** 2 + 105 * (ep * r) ** 3 + 15 * (ep * r) ** 4 + 1 * (
                            ep * r) ** 5)
        elif k == 6:
            self.name = 'mat6'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (
                        10395 + 10395 * ep * r + 4725 * (ep * r) ** 2 + 1260 * (ep * r) ** 3 + 210 * (
                            ep * r) ** 4 + 21 * (ep * r) ** 5 + 1 * (ep * r) ** 6)
        elif k == 7:
            self.name = 'mat7'
            self.rbf = lambda ep, r: torch.exp(-ep * r) * (
                        135135 + 135135 * ep * r + 62370 * (ep * r) ** 2 + 17325 * (ep * r) ** 3 + 3150 * (
                            ep * r) ** 4 + 378 * (ep * r) ** 5 + 28 * (ep * r) ** 6 + 1 * (ep * r) ** 7)
        else:
            self.name = None
            self.rbf = None
            raise Exception('This Matern kernel is not implemented')


class Gaussian(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.name = 'gauss'
        self.rbf = lambda ep, r: torch.exp(-(ep * r) ** 2)