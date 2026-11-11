# RF Signal Workbench

A Python-based learning and experimentation workbench for **RF and digital signal processing**. Generate common signals, add controlled AWGN, inspect time- and frequency-domain behavior, apply digital filters, and explore basic AM/FM modulation through an interactive Streamlit interface.

> This is an educational engineering project, not a certified RF measurement instrument. Numerical results depend on the selected sampling and simulation parameters.

## Features

- Generate sine, square, triangle, and composite signals
- Add additive white Gaussian noise (AWGN) with a configurable SNR
- Inspect waveform and FFT magnitude spectrum
- Calculate peak amplitude, RMS, dominant frequency, and estimated SNR
- Apply Butterworth low-pass, high-pass, and band-pass filters
- Explore basic amplitude and frequency modulation
- Export generated samples to CSV

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
python -m pytest
```

## Project layout

```text
RF-Signal-Workbench/
├── app.py
├── rfworkbench/
│   ├── __init__.py
│   ├── signals.py
│   ├── analysis.py
│   ├── filters.py
│   └── modulation.py
├── tests/
│   ├── test_signals.py
│   └── test_analysis.py
├── requirements.txt
└── pyproject.toml
```

## Engineering notes

- Frequencies are expressed in hertz (Hz), time in seconds (s), and amplitudes in linear units.
- The sampling frequency must exceed twice the highest frequency of interest (Nyquist criterion); practical simulations should leave additional margin.
- FFT plots use a one-sided amplitude spectrum for real-valued signals.
- Filter cutoff frequencies must be below the Nyquist frequency.

## Roadmap

- Add unit-aware parameter validation and clearer error messages
- Add filter response plots and group-delay exploration
- Add modulation spectrum and demodulation experiments
- Add reproducible example scenarios and CI test automation

## License

MIT License. See [LICENSE](LICENSE).
