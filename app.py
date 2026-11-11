"""Streamlit interface for the RF Signal Workbench."""

import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from rfworkbench.signals import generate_signal, add_awgn
from rfworkbench.analysis import analyze_signal, spectrum
from rfworkbench.filters import apply_filter
from rfworkbench.modulation import am_modulate, fm_modulate


def _csv_bytes(time_values: np.ndarray, signal_values: np.ndarray) -> bytes:
    """Serialize time and amplitude columns as UTF-8 CSV bytes."""
    buffer = io.StringIO()
    np.savetxt(buffer, np.column_stack((time_values, signal_values)), delimiter=",", header="time_s,amplitude", comments="")
    return buffer.getvalue().encode("utf-8")


st.set_page_config(page_title="RF Signal Workbench", page_icon="📡", layout="wide")
st.title("📡 RF Signal Workbench")
st.caption("An interactive sandbox for RF and digital signal-processing fundamentals.")

with st.sidebar:
    st.header("Signal setup")
    kind = st.selectbox("Waveform", ["sine", "square", "triangle"])
    fs = st.number_input("Sampling frequency (Hz)", min_value=100.0, max_value=200_000.0, value=10_000.0, step=100.0)
    duration = st.number_input("Duration (seconds)", min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    frequency = st.number_input("Signal frequency (Hz)", min_value=1.0, max_value=float(fs / 2 - 1), value=min(250.0, float(fs / 2 - 1)), step=10.0)
    amplitude = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)
    add_noise = st.checkbox("Add AWGN")
    snr_db = st.slider("Target SNR (dB)", -5, 40, 20) if add_noise else 20
    seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

try:
    t, clean = generate_signal(kind, frequency, fs, duration, amplitude)
    noisy = add_awgn(clean, float(snr_db), np.random.default_rng(int(seed))) if add_noise else clean.copy()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Signal analysis")
metrics = analyze_signal(noisy, fs)
cols = st.columns(4)
cols[0].metric("Peak amplitude", f"{metrics['peak']:.3f}")
cols[1].metric("RMS", f"{metrics['rms']:.3f}")
cols[2].metric("Dominant frequency", f"{metrics['dominant_frequency_hz']:.1f} Hz")
cols[3].metric("Estimated SNR", "∞" if not np.isfinite(metrics["estimated_snr_db"]) else f"{metrics['estimated_snr_db']:.1f} dB")

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, noisy, linewidth=1, label="Observed")
    if add_noise:
        ax.plot(t, clean, linewidth=1, alpha=0.7, label="Clean reference")
    ax.set(title="Time domain", xlabel="Time (s)", ylabel="Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
with right:
    f, mag = spectrum(noisy, fs)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(f, mag, linewidth=1)
    ax.set(title="One-sided amplitude spectrum", xlabel="Frequency (Hz)", ylabel="Amplitude")
    ax.set_xlim(0, fs / 2)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Digital filtering")
filter_mode = st.selectbox("Filter type", ["lowpass", "highpass", "bandpass"])
if filter_mode == "bandpass":
    c1, c2 = st.columns(2)
    low = c1.number_input("Lower cutoff (Hz)", min_value=1.0, max_value=float(fs / 2 - 2), value=min(100.0, float(fs / 2 - 2)))
    high = c2.number_input("Upper cutoff (Hz)", min_value=2.0, max_value=float(fs / 2 - 1), value=min(1000.0, float(fs / 2 - 1)))
    cutoff = (low, high)
else:
    cutoff = st.number_input("Cutoff frequency (Hz)", min_value=1.0, max_value=float(fs / 2 - 1), value=min(500.0, float(fs / 2 - 1)))
order = st.slider("Butterworth order", 1, 10, 4)
if st.button("Apply filter"):
    try:
        filtered = apply_filter(noisy, fs, filter_mode, cutoff, order)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(t, noisy, alpha=0.45, label="Input")
        ax.plot(t, filtered, label="Filtered")
        ax.set(xlabel="Time (s)", ylabel="Amplitude", title=f"{filter_mode.title()} response")
        ax.grid(True, alpha=0.25)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.download_button("Download filtered samples (CSV)", _csv_bytes(t, filtered), "filtered_signal.csv", "text/csv")
    except ValueError as exc:
        st.error(str(exc))

st.subheader("AM / FM modulation")
mod_col1, mod_col2 = st.columns(2)
with mod_col1:
    carrier = st.number_input("Carrier frequency (Hz)", min_value=1.0, max_value=float(fs / 2 - 1), value=min(1500.0, float(fs / 2 - 1)))
    modulation_index = st.slider("AM modulation index", 0.0, 1.0, 0.5, 0.05)
    deviation = st.number_input("FM frequency deviation (Hz)", min_value=0.0, max_value=float(fs / 4), value=min(200.0, float(fs / 4)))
    mod_type = st.radio("Modulation", ["AM", "FM"], horizontal=True)
with mod_col2:
    try:
        modulated = am_modulate(t, clean, carrier, modulation_index) if mod_type == "AM" else fm_modulate(t, clean, carrier, deviation)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(t, modulated, linewidth=1)
        ax.set(title=f"{mod_type} waveform", xlabel="Time (s)", ylabel="Amplitude")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
    except ValueError as exc:
        st.warning(str(exc))

st.download_button("Download current signal (CSV)", _csv_bytes(t, noisy), "signal_samples.csv", "text/csv")
st.caption("Educational simulation only. Check sampling, units, and model assumptions before interpreting results.")
