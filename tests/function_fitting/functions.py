import math

import numpy as np


def example_function(config):
    x = np.linspace(config.min, config.max, config.samples)
    function = {
        "Gaussian": _gaussian,
        "Constant": _constant,
        "Sawtooth": _sawtooth,
    }[config.function]
    return function(config, x)


def _gaussian(config, x):
    mean = config.gauss_mean
    sigma = config.gauss_sigma
    # apply function
    f = (
        1
        / (sigma * math.sqrt(2.0 * math.pi))
        * np.exp(-1 / 2.0 * ((x - mean) / sigma) ** 2)
    )
    f = 1 / float(max(f)) * f
    # return
    return f


def _sawtooth(config, x):
    # apply function
    f = np.ones_like(x)
    f[::2] = 0.0
    # return
    return f


def _constant(config, x):
    # apply function
    f = np.ones_like(x)
    f[int(len(f) / 2) :] = -1.0
    # return
    return f
