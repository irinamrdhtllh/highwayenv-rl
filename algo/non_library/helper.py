import numpy as np
import scipy.signal as signal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
