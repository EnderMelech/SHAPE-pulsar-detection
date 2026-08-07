import numpy as np
def W(n):
    return np.exp(j*2*np.pi/n)
def add_zeros(x):
    return x.extend([0]*(len(x)-2**len(x).bit_length()))

def dft_helper(x, y):
    if len(x) == 1:
        return x
    if set(x) == {0}:
        return x
    n = len(x)
    x_1 = x[::2]
    x_2 = x[1::2]

    return [dft_helper(x_1)[k%(n//2)]  + \
        W(len(x)//2)**(-k)*dft_helper(x_2)[k%(n//2)] \
            for k in range(n)]
    #return merge(dft(x[::2]) , dft(x[1::2])
