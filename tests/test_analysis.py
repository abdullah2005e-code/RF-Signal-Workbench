import numpy as np
from rfworkbench.analysis import analyze_signal, spectrum


def test_spectrum_finds_dominant_tone():
    fs = 10_000
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 250 * t)
    f, a = spectrum(x, fs)
    assert abs(f[np.argmax(a)] - 250) < 2
    metrics = analyze_signal(x, fs)
    assert abs(metrics["dominant_frequency_hz"] - 250) < 2
    assert np.isclose(metrics["rms"], 1 / np.sqrt(2), atol=0.01)
