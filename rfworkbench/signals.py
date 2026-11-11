"""Signal generation and additive white Gaussian noise utilities."""

import numpy as np
from scipy import signal


def _validate_sampling(fs: float, duration: float) -> None:
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be a positive finite number.")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be a positive finite number.")


def generate_signal(kind: str, frequency: float, fs: float, duration: float,
                    amplitude: float = 1.0, phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return time and signal arrays. Phase is specified in radians."""
    _validate_sampling(fs, duration)
    if not np.isfinite(frequency) or frequency <= 0 or frequency >= fs / 2:
        raise ValueError("Frequency must be positive and below the Nyquist frequency.")
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Amplitude must be finite and non-negative.")
    if not np.isfinite(phase):
        raise ValueError("Phase must be finite.")
    count = max(2, int(round(fs * duration)))
    t = np.arange(count, dtype=float) / fs
    angle = 2 * np.pi * frequency * t + phase
    if kind == "sine":
        y = amplitude * np.sin(angle)
    elif kind == "square":
        y = amplitude * signal.square(angle)
    elif kind == "triangle":
        y = amplitude * signal.sawtooth(angle, width=0.5)
    else:
        raise ValueError("kind must be 'sine', 'square', or 'triangle'.")
    return t, y.astype(float)


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add real white Gaussian noise at the requested measured-power SNR."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Input must be a non-empty, finite one-dimensional signal.")
    if not np.isfinite(snr_db):
        raise ValueError("SNR must be finite.")
    power = float(np.mean(x ** 2))
    if power == 0:
        raise ValueError("Cannot set SNR for a zero-power signal.")
    generator = rng if rng is not None else np.random.default_rng()
    noise_power = power / (10 ** (snr_db / 10))
    noise = generator.normal(0.0, np.sqrt(noise_power), size=x.size)
    return x + noise
