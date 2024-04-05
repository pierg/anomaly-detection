"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch

from anomaly_detection.optimizers.base import Optimizer


class AdamW(Optimizer):
    """
    Implements AdamW optimizer, a variant of the Adam optimizer with weight decay.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        Initializes the AdamW optimizer.

        :param params: Iterable of parameters to optimize. Should be an iterable of `torch.Tensor`s.
        :param lr: Learning rate.
        :param betas: Coefficients used for computing running averages of gradient and its square.
        :param eps: Term added to the denominator to improve numerical stability.
        :param weight_decay: Weight decay coefficient.
        """
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (
                param.grad**2
            )

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            # Update parameters
            param.data.addcdiv_(m_hat, (v_hat.sqrt() + self.eps), value=-self.lr)

            # Apply weight decay
            if self.weight_decay != 0:
                param.data.add_(param.data, alpha=-self.lr * self.weight_decay)
