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
    a = input("here")
    if a != 'd':
        a = [float(i) for i in a.split()]
        x = np.array(a, dtype=np.complex128)
    else:
        x = np.arange(8, dtype=np.complex128)
    print(fft(x))