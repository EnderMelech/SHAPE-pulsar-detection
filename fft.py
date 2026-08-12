import numpy as np


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
        from pulsar_code import get_best_time_series
    except ImportError:
        raise ImportError(
            "Cannot import get_best_time_series from pulsar_code. "
            "Make sure pulsar_code.py is in the same folder."
        )

    best_ts, _, _, _ = get_best_time_series()
    best_ts = np.asarray(best_ts, dtype=np.complex128)
    fft_result = fft(best_ts)
    print(fft_result)