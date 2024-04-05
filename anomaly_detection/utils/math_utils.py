"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

# """
# Author: Piergiuseppe Mallozzi
# Date: 2024
# """

# import numpy as np

# def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
#     """
#     Compute the softmax of each row or column of the input x.

#     :param x: Numpy array of logits.
#     :param axis: Axis along which the softmax operation is applied.
#     :return: Softmax applied array.
#     """
#     e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
#     return e_x / e_x.sum(axis=axis, keepdims=True)


# def multinomial(probs: np.ndarray, num_samples: int = 1) -> np.ndarray:
#     """
#     Sample indices from a multinomial distribution.

#     :param probs: Probability distribution array.
#     :param num_samples: Number of samples to draw.
#     :return: Indices sampled according to the given probability distribution.
#     """
# return np.array([np.random.choice(len(p), size=num_samples, p=p) for p
# in probs])

# def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
#     """
#     Compute the cross-entropy loss.

#     :param logits: Logits array.
#     :param targets: Target indices.
#     :return: Cross-entropy loss.
#     """
#     m = targets.shape[0]
#     log_softmax_logits = np.log(softmax(logits))
#     # Pick the logits for the correct labels for each example in the batch
#     correct_logits = log_softmax_logits[np.arange(m), targets]
#     # Compute the mean negative log probability
#     loss = -np.mean(correct_logits)
#     return loss
