import numpy as np
import time

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
        print("imported pulsar_code.py")
    except ImportError:
        raise ImportError(
            "Cannot import get_best_time_series from pulsar_code. "
            "Make sure pulsar_code.py is in the same folder."
        )

    print("get_best_time_series called")
    start_time = time.perf_counter()
    result = get_best_time_series(timing=False, plot_fft=True)
    end_time = time.perf_counter()
    print(f"get_best_time_series finished in {end_time - start_time:.6f} seconds")
    best_ts = result[0] # type: ignore
    print("assigned first element of result to best_ts")
    start_time = time.perf_counter()
    best_ts = np.asarray(best_ts, dtype=np.complex128)
    end_time = time.perf_counter()
    print(f"converted best_ts to a comples numpy array in {end_time - start_time:.6f} seconds")
    print("started custom fft")
    start_time = time.perf_counter()
    fft_result = fft(best_ts)
    end_time = time.perf_counter()
    print(f"finished custom fft in {end_time - start_time:.6f} seconds")
    print("started np.fft")
    start_time = time.perf_counter()
    np_result = np.fft.fft(best_ts)
    end_time = time.perf_counter()
    print(f"finished np.fft in {end_time - start_time:.6f} seconds")
    print(np_result)
    print(fft_result)
    print(np.allclose(fft_result, np_result))
    print(np.max(np.abs(fft_result - np_result)))  # see how small the actual error is