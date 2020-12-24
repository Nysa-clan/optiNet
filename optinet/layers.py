"""
layers
"""
import numpy as np
from optinet.tensor import Tensor


class Layer:
    def __init__(self):
        pass

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Make the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagation
        """
        raise NotImplementedError
