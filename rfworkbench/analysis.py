"""Time-domain metrics and one-sided FFT analysis."""

import numpy as np


def _as_signal(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Signal must be a finite one-dimensional array with at least two samples.")
    return values


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided frequency bins and amplitude spectrum."""
    values = _as_signal(x)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be positive and finite.")
    n = values.size
    window = np.hanning(n)
    coherent_gain = np.sum(window) / n
    if coherent_gain == 0:
        raise ValueError("Unable to calculate spectrum for this signal length.")
    yf = np.fft.rfft(values * window)
    amplitude = np.abs(yf) / (n * coherent_gain)
    if n > 2:
        amplitude[1:-1 if n % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(n, d=1 / fs), amplitude


def analyze_signal(x: np.ndarray, fs: float) -> dict[str, float]:
    """Calculate peak, RMS, dominant non-DC frequency and estimated SNR."""
    values = _as_signal(x)
    freqs, amps = spectrum(values, fs)
    peak_index = int(np.argmax(amps[1:]) + 1) if amps.size > 1 else 0
    centered = values - np.mean(values)
    total_power = float(np.mean(centered ** 2))
    dominant_power = float((amps[peak_index] ** 2) / 2) if peak_index else 0.0
    residual = max(total_power - dominant_power, 0.0)
    snr = float(10 * np.log10(dominant_power / residual)) if residual > 0 and dominant_power > 0 else float("inf")
    return {
        "peak": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "dominant_frequency_hz": float(freqs[peak_index]),
        "estimated_snr_db": snr,
    }
