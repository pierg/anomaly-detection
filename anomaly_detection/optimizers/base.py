"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch

class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, params, lr=0.001):
        """
        Initializes the optimizer.

        :param params: Iterable of parameters to optimize. Should be an iterable of `torch.Tensor`s.
        :param lr: Learning rate.
        """
        self.params = list(params)
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def zero_grad(self):
        """
        Zeros the gradient of all the parameters.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
