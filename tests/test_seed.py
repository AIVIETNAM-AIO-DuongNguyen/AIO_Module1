import random

import numpy as np
import torch

from src.utils.seed import make_generator, set_seed


def test_set_seed_is_reproducible():
    set_seed(123)
    first = (random.random(), float(np.random.rand()), torch.rand(1).item())
    set_seed(123)
    second = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert first == second


def test_make_generator_is_deterministic():
    gen_a = make_generator(7)
    gen_b = make_generator(7)
    assert torch.equal(torch.rand(4, generator=gen_a), torch.rand(4, generator=gen_b))
