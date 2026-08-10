"""
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
"""

def fft(a):
    # if a is not a power of 2 append zeros until it is
    # n = length(a)

    # base cases
    # if n == 2 skip the recursive calls and go straight to the butterfly thing
    # return a if n == 1
    # return 0 if every item in a is 0

    # make an array of all the even-indexed items in a
    # make an array of all the odd-indexed items in a
    
    # assign a variable the result of the recursive call with the even array
        # e.g. result_even = fft(a_even)
    # assign a variable the result of the recursive call with the odd array
# this is binary tree recursion thing
    # precompute array of size n
    # do butterfly thingamabob all the legit calculations
    # do it for conjugate too
    # insert into array
    # do it for every k
    # return final array