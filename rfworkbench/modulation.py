"""Basic AM and FM modulation examples."""

import numpy as np


def _validate(t: np.ndarray, message: np.ndarray, carrier_hz: float) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(t, dtype=float)
    msg = np.asarray(message, dtype=float)
    if time.ndim != 1 or msg.ndim != 1 or time.size != msg.size or time.size < 2:
        raise ValueError("Time and message must be one-dimensional arrays of equal length.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(msg)):
        raise ValueError("Time and message must contain only finite values.")
    if not np.isfinite(carrier_hz) or carrier_hz <= 0:
        raise ValueError("Carrier frequency must be positive and finite.")
    return time, msg


def am_modulate(t: np.ndarray, message: np.ndarray, carrier_hz: float, modulation_index: float = 0.5) -> np.ndarray:
    """Conventional AM with a carrier; modulation index should be 0..1."""
    time, msg = _validate(t, message, carrier_hz)
    if not np.isfinite(modulation_index) or not 0 <= modulation_index <= 1:
        raise ValueError("Modulation index must be between 0 and 1 for this conventional AM model.")
    peak = float(np.max(np.abs(msg)))
    normalized = msg / peak if peak > 0 else np.zeros_like(msg)
    return (1 + modulation_index * normalized) * np.cos(2 * np.pi * carrier_hz * time)


def fm_modulate(t: np.ndarray, message: np.ndarray, carrier_hz: float, frequency_deviation_hz: float) -> np.ndarray:
    """Generate FM by integrating instantaneous frequency."""
    time, msg = _validate(t, message, carrier_hz)
    if not np.isfinite(frequency_deviation_hz) or frequency_deviation_hz < 0:
        raise ValueError("Frequency deviation must be finite and non-negative.")
    if not np.all(np.diff(time) > 0):
        raise ValueError("Time samples must be strictly increasing.")
    peak = float(np.max(np.abs(msg)))
    normalized = msg / peak if peak > 0 else np.zeros_like(msg)
    phase = 2 * np.pi * carrier_hz * time + 2 * np.pi * frequency_deviation_hz * np.cumsum(normalized) * np.mean(np.diff(time))
    return np.cos(phase)
