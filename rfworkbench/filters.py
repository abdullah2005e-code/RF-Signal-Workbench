"""Butterworth digital filter helpers."""

import numpy as np
from scipy import signal


def apply_filter(x: np.ndarray, fs: float, mode: str, cutoff_hz: float | tuple[float, float], order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth low/high/band-pass filter."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size < 4 or not np.all(np.isfinite(values)):
        raise ValueError("Input must be a finite one-dimensional signal with at least four samples.")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be positive and finite.")
    if not isinstance(order, int) or order < 1:
        raise ValueError("Filter order must be a positive integer.")
    nyquist = fs / 2
    if mode in ("lowpass", "highpass"):
        if not np.isscalar(cutoff_hz) or not 0 < float(cutoff_hz) < nyquist:
            raise ValueError("Cutoff must lie between 0 and the Nyquist frequency.")
        wn = float(cutoff_hz)
    elif mode == "bandpass":
        if not isinstance(cutoff_hz, (tuple, list)) or len(cutoff_hz) != 2:
            raise ValueError("Band-pass cutoff must be a (low, high) pair.")
        low, high = map(float, cutoff_hz)
        if not 0 < low < high < nyquist:
            raise ValueError("Band-pass cutoffs must satisfy 0 < low < high < Nyquist.")
        wn = (low, high)
    else:
        raise ValueError("mode must be 'lowpass', 'highpass', or 'bandpass'.")
    sos = signal.butter(order, wn, btype=mode, fs=fs, output="sos")
    return signal.sosfiltfilt(sos, values)
