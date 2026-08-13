import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

plt.ioff()

def fft(data):
    n = data.shape[0]
    if n <= 1:
        return data

    twiddle_cache = {
        size: np.exp(-2j * np.pi * np.arange(size // 2) / size)
        for size in (2 ** k for k in range(2, int(np.log2(n)) + 1))
    }

    return fft_butterfly(data, twiddle_cache)


def fft_butterfly(data, twiddle_cache):
    n = data.shape[0]
    if n == 2:
        return np.array([data[0] + data[1], data[0] - data[1]], dtype=np.complex128)

    evens = fft_butterfly(data[::2], twiddle_cache)
    odds = fft_butterfly(data[1::2], twiddle_cache)

    twiddles = twiddle_cache[n]
    upper = evens + twiddles * odds
    lower = evens - twiddles * odds
    return np.concatenate([upper, lower])

if __name__ == "__main__":
    try:
        with open(f"s120408_215426_gbts.pkl", "rb") as f:
            fft_ready_series, n_real, best_dedispersion, tbin, freqs, best_dm, filename = pickle.load(f)
        print("loaded gbts data") # gbts means get_best_time_series()
    except FileNotFoundError:
        print("gbts data not found")
        try:
            from pulsar_code import get_best_time_series
            print("imported pulsar_code.py")
        except ImportError:
            raise ImportError(
                "Cannot import get_best_time_series from pulsar_code. "
                "Make sure pulsar_code.py is in the same folder."
            )

        print("get_best_time_series called")
        start_time = time.perf_counter()
        gbts = fft_ready_series, n_real, best_dedispersion, tbin, freqs, best_dm, filename = get_best_time_series(timing=False)
        end_time = time.perf_counter()
        print(f"get_best_time_series finished in {end_time - start_time:.6f} seconds")

        with open(f"{filename}_gbts.pkl", "wb") as f:
            pickle.dump(gbts, f)

    best_ts = fft_ready_series  # the FFT-ready (padded/tapered) time series
    print("received FFT-ready time series from get_best_time_series")

    # Prepare pre-FFT plotting data but don't display yet; show at end.
    try:
        pre_ts = best_ts.real if np.iscomplexobj(best_ts) else np.asarray(best_ts, dtype=float)
    except Exception:
        pre_ts = None

    start_time = time.perf_counter()
    best_ts = np.asarray(best_ts, dtype=np.complex128)
    end_time = time.perf_counter()
    print(f"converted best_ts to a complex numpy array in {end_time - start_time:.6f} seconds")
    print("started custom fft")
    start_time = time.perf_counter()
    fft_result = fft(best_ts)
    end_time = time.perf_counter()
    print(f"finished custom fft in {end_time - start_time:.6f} seconds")

    # Prepare post-FFT plotting data but don't display yet; show at end.
    try:
        N = len(fft_result)
        freqs_fft = np.fft.fftfreq(N, d=tbin)
        power = np.abs(fft_result) ** 2
        mask = freqs_fft >= 0
        freqs_fft_masked = freqs_fft[mask]
        power_masked = power[mask]
        max_freq = float(np.max(freqs_fft_masked)) if freqs_fft_masked.size else 1.0
        freq_cap = min(max_freq, 25.0)
    except Exception:
        freqs_fft_masked = None
        power_masked = None
        freq_cap = 250.0
    print("started np.fft")
    start_time = time.perf_counter()
    np_result = np.fft.fft(best_ts)
    end_time = time.perf_counter()
    print(f"finished np.fft in {end_time - start_time:.6f} seconds")
    print(np_result)
    print(fft_result)
    print(np.allclose(fft_result, np_result))
    print(np.max(np.abs(fft_result - np_result)))  # see how small the actual error is

    # --- Show time series first, wait for Enter key in the figure window,
    # then show FFT and wait for Enter there as well. This avoids terminal
    # focus issues.
    try:
        if pre_ts is not None:
            fig1, ax1 = plt.subplots(figsize=(10, 3))
            ax1.plot(pre_ts)
            ax1.set_title("Time series returned by get_best_time_series (pre-FFT)")
            ax1.set_xlabel("Sample")
            ax1.set_ylabel("Amplitude")
            fig1.tight_layout()

            def _on_key_pre(event):
                if getattr(event, 'key', None) == 'enter':
                    plt.close(fig1)

            fig1.canvas.mpl_connect('key_press_event', _on_key_pre)
            # Blocking show until user presses Enter (in the figure)
            plt.show(block=True)

        if freqs_fft_masked is not None and power_masked is not None:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(freqs_fft_masked, power_masked)
            ax2.set_title("Custom FFT Power Spectrum (post-FFT)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Power")
            ax2.set_xlim(0, freq_cap)
            # mark ticks every 5 Hz
            try:
                ticks = np.arange(0.0, freq_cap + 0.001, 5.0)
                ax2.set_xticks(ticks)
            except Exception:
                pass
            fig2.tight_layout()

            def _on_key_fft(event):
                if getattr(event, 'key', None) == 'enter':
                    plt.close(fig2)

            fig2.canvas.mpl_connect('key_press_event', _on_key_fft)
            # Blocking show until user presses Enter (in the figure)
            plt.show(block=True)
    except Exception as e:
        print("final plotting failed:", e)