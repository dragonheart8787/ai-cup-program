import random

import numpy as np
import pytest

from src.utils.seed import seed_everything


def test_python_and_numpy_reproducible():
    seed_everything(123)
    a = (random.random(), np.random.rand(3).tolist())
    seed_everything(123)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_different_seed_differs():
    seed_everything(1)
    a = np.random.rand(3).tolist()
    seed_everything(2)
    assert a != np.random.rand(3).tolist()


def test_torch_reproducible_if_installed():
    torch = pytest.importorskip("torch")
    seed_everything(7)
    a = torch.rand(3)
    seed_everything(7)
    assert torch.equal(a, torch.rand(3))
