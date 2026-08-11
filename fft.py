import numpy as np
def fft_butterfly(data):
    n = len(data)

    if n == 2:
        return [data[0] + data[1], data[0] - data[1]]

    data_evens = data[::2]
    data_odds = data[1::2]

    result_even = fft(data_evens)
    result_odd = fft(data_odds)

    result = [0] * n

    for k in range(n // 2):
        w = np.exp(-2 * np.pi * 1j * k / n)
        t = w * result_odd[k]
        result[k] = result_even[k] + t
        result[k + n // 2] = result_even[k] - t
    return result

def fft(a):
    data = og_data.copy()
    if 


data = [float(x) for x in input("Enter numbers separated by spaces: ").split()]
fft(data)
# print([round(float(x), 5) for x in np.abs(fft(data))])