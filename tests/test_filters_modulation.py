import numpy as np
import pytest
from rfworkbench.filters import apply_filter
from rfworkbench.modulation import am_modulate, fm_modulate


def test_lowpass_reduces_high_frequency_tone():
    fs = 10_000
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1500 * t)
    y = apply_filter(x, fs, "lowpass", 300, order=5)
    assert np.std(y - np.sin(2 * np.pi * 100 * t)) < 0.15


def test_invalid_bandpass_is_rejected():
    with pytest.raises(ValueError):
        apply_filter(np.ones(100), 1000, "bandpass", (300, 200))


def test_am_output_has_same_length_and_bounded_envelope():
    t = np.arange(1000) / 10_000
    message = np.sin(2 * np.pi * 10 * t)
    y = am_modulate(t, message, 1000, 0.7)
    assert y.shape == message.shape
    assert np.max(np.abs(y)) <= 1.71


def test_fm_output_is_finite():
    t = np.arange(1000) / 10_000
    message = np.sin(2 * np.pi * 10 * t)
    y = fm_modulate(t, message, 1000, 100)
    assert y.shape == message.shape
    assert np.all(np.isfinite(y))
