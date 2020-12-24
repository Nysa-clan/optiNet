"""
A loss functions
"""

import numpy as np
from optinet.tensor import Tensor


class Loss:
    """
    Loss class
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    Mean Squred Error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)


class MAE(Loss):
    """
    Mean Absolute Error
    """
    pass


class BinaryCrossEntropy(Loss):
    """
    Implementation of BinaryCrossEntropy
    """
    pass
