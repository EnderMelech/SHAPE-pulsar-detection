# MAKE SURE TO RUN THIS FIRST BEFORE FOURIER_VISUALIZTION!!!!
# INSTRUCTIONS ON HOW TO RUN THE VISUALIZAITON IN ITS FILE
# to generate the data necessary for the video do this:
"""
ctrl + `
cd manim_animations
python fft_data_bridge.py
"""
# also somewhere there is an ADD_NOISE boolean which you can turn on if you want to make the data a bit more random

"""
Runs your custom radix-2 Cooley-Tukey FFT and packages the results into a
.npz file that the Manim scene (fourier_visualization.py) can load.

Why this exists: your raw signal (radio telescope data) can be huge, but a
Manim animation only needs to *draw* a couple thousand points and a handful
of dominant sine components. This script does the heavy numerical work
once, ahead of time, so the animation itself stays fast to render.

Usage:
    python fft_data_bridge.py
(or import prepare_fft_visualization_data() into your own driver script)
"""

import os
import sys
import numpy as np

# fft.py lives one directory up (repo root), while this file lives inside
# manim animations/. Add the parent directory to sys.path based on this
# file's own location -- not the current working directory -- so the
# import below works no matter where you run this script from.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fft import fft as fft_func
# --------------------------------------------------------------


def prepare_fft_visualization_data(
    signal: np.ndarray,
    sample_rate: float,
    fft_func,
    n_components: int = 6,
    output_path: str = "fft_viz_data.npz",
    max_plot_points: int = 2000,
):
    """
    Parameters
    ----------
    signal : np.ndarray (complex128)
        Your input signal. For radio/pulsar data this is often I/Q
        (complex baseband) data; if your signal is real-valued, cast it
        to complex first (signal.astype(np.complex128)) since your FFT
        expects complex float input/output.
    sample_rate : float
        Sampling rate in Hz.
    fft_func : callable
        Your custom radix-2 Cooley-Tukey FFT. Signature:
        fft_func(np.ndarray[complex128]) -> np.ndarray[complex128]
    n_components : int
        Number of dominant *distinct* frequency components to pull out
        for the "splitting into individual waves" part of the animation.
        For real-valued signals, +f/-f conjugate pairs are collapsed into
        a single real sinusoid (see Fix 4 below), so this is the number
        of physically distinct tones shown, not raw FFT bins.
    output_path : str
        Where to save the bundle Manim will read.
    max_plot_points : int
        Minimum time-domain points kept for drawing the original
        waveform. This is a rendering-only downsample -- the FFT itself
        still runs on the full-resolution signal you pass in. The actual
        number used may be higher than this if needed to avoid aliasing
        (see Fix 3 below).
    """
    signal = np.asarray(signal, dtype=np.complex128)
    n = len(signal)
    assert n > 0 and (n & (n - 1)) == 0, (
        f"Signal length must be a power of 2 for radix-2 FFT, got {n}. "
        "Zero-pad your data up to the next power of 2 first."
    )

    spectrum = fft_func(signal)
    assert spectrum.shape == signal.shape, "FFT output shape mismatch"

    # np.fft.fftfreq matches standard FFT bin ordering (0, +f, ..., -f, ...)
    # which is what a standard Cooley-Tukey FFT produces.
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # --- Fix 4: collapse +f/-f conjugate pairs into one real component ---
    # For a real-valued signal, the +f and -f bins are a mirror pair
    # representing ONE physical oscillation, not two separate tones.
    # Walk bins in descending magnitude order and keep only the first
    # bin seen for each |frequency|.
    order = np.argsort(magnitude)[::-1]

    seen_abs_freqs = set()
    top_bins = []
    for idx in order:
        # ensure a native Python float is passed to round() to satisfy type checkers
        f_key = round(float(abs(freqs[idx])), 6)
        if f_key in seen_abs_freqs:
            continue
        seen_abs_freqs.add(f_key)
        top_bins.append(idx)
        if len(top_bins) == n_components:
            break
    top_bins = np.array(top_bins)

    component_freqs = freqs[top_bins]
    component_amps = magnitude[top_bins] / n  # normalize to true amplitude
    component_phases = phase[top_bins]

    # Each kept bin represents a conjugate pair collapsed into one real
    # sinusoid, so double its amplitude to match its true contribution to
    # the original signal (skip this for f=0 / Nyquist, which have no pair).
    for i, f in enumerate(component_freqs):
        if not np.isclose(f, 0.0) and not np.isclose(abs(f), sample_rate / 2):
            component_amps[i] *= 2

    # --- Fix 3: make sure we plot enough points to avoid aliasing ---
    # Naively striding down to a fixed point count can undersample fast
    # oscillations and make the waveform look like solid noise instead of
    # a smooth curve. Make sure we keep enough points to resolve roughly
    # 15 samples per cycle of the fastest component actually present.
    duration = n / sample_rate
    max_freq_present = np.abs(freqs).max()
    min_points_needed = int(duration * max_freq_present * 15)
    effective_max_plot_points = max(max_plot_points, min(n, min_points_needed))

    if n > effective_max_plot_points:
        stride = n // effective_max_plot_points
        t = np.arange(0, n, stride) / sample_rate
        signal_plot = signal[::stride]
    else:
        t = np.arange(n) / sample_rate
        signal_plot = signal

    np.savez(
        output_path,
        t=t,
        signal_real=np.real(signal_plot),
        signal_imag=np.imag(signal_plot),
        freqs=freqs,
        magnitude=magnitude,
        component_freqs=component_freqs,
        component_amps=component_amps,
        component_phases=component_phases,
        sample_rate=sample_rate,
        duration=duration,
    )
    print(f"Saved visualization data to {output_path}")
    print(f"Top {n_components} components (Hz, amplitude):")
    for f, a in zip(component_freqs, component_amps):
        print(f"  {f:10.3f} Hz   amp={a:.4f}")
    return output_path


if __name__ == "__main__":
    # --- Example / placeholder. Swap in your real signal + fft_func. ---
    N = 128            # number of samples (must be power of 2)
    sample_rate = 64.0  # Hz
    t = np.linspace(0, (N - 1) / sample_rate, N)
    F1, F2 = 5.0, 12.0   # keep both under Nyquist (sample_rate / 2)

    ADD_NOISE = False   # flip to True for a less "clean" looking signal
    noise = np.random.normal(0, 0.05, N) if ADD_NOISE else 0.0

    # Toy superposed signal: replace with your actual radio data.
    signal = (
        np.sin(2 * np.pi * F1 * t) + 0.6 * np.sin(2 * np.pi * F2 * t) + noise
    ).astype(np.complex128)

    prepare_fft_visualization_data(
        signal=signal,
        sample_rate=sample_rate,
        fft_func=fft_func,
        n_components=2,
        output_path="fft_viz_data.npz",
    )