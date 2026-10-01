import random

import numpy as np
import torch

from oncolens.utils.seed import rank_seed, seed_everything


def draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_same_numbers():
    seed_everything(123)
    a = draw()
    seed_everything(123)
    assert draw() == a


def test_ranks_get_different_streams():
    seed_everything(0, rank=0)
    a = draw()
    seed_everything(0, rank=1)
    assert draw() != a
    assert rank_seed(0, 1) == 1
