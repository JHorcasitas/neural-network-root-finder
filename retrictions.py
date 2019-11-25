from operator import mul
from functools import reduce
from itertools import combinations

import numpy as np


# TODO: make this readable
def get_F(W, i):
    W_list = list(W)
    del W_list[i]
    F = np.empty((1, len(W_list) + 1), dtype=np.float32)
    F[0, 0] = -1
    F[0, 1] = sum(W_list)
    for i in range(2, len(W_list) + 1):
        p = sum(product(c, W_list) for c in combinations(range(len(W_list)), i))
        F[0, i] = (-1) ** (i + 1) * p
    return F


def product(indices, W):
    W = [W[idx] for idx in indices]
    return reduce(mul, W, 1)
