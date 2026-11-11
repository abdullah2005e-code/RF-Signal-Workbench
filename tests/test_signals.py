import numpy as np
import pytest
from rfworkbench.signals import generate_signal, add_awgn


def test_sine_generation_has_expected_shape_and_amplitude():
    t, y = generate_signal("sine", 100, 10_000, 0.1, amplitude=2)
    assert t.shape == y.shape
    assert len(y) == 1000
    assert np.isclose(np.max(y), 2, atol=0.01)


def test_invalid_frequency_is_rejected():
    with pytest.raises(ValueError):
        generate_signal("sine", 5_000, 10_000, 0.1)


def test_awgn_is_reproducible_with_generator():
    x = np.ones(1000)
    a = add_awgn(x, 10, np.random.default_rng(7))
    b = add_awgn(x, 10, np.random.default_rng(7))
    assert np.allclose(a, b)
