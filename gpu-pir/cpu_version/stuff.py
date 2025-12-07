from math import ceil, log
from random import randint

from params import *

def to_words(k, B, n_words):
    return [(k // B**i) % B for i in range(n_words)]

def from_words(t_words, B):
    return sum([x * (B**i) for i, x in enumerate(t_words)])

def to_mont(x, B, N):
    p = int(ceil(log(N, B)))
    r = p
    R = B**r
    return (x * R) % N

# T < RN, it is the wide product.
# If you had double montgomery I don't see why you couldn't just add to this a * b + c
# Can remove with ICRT and insert with first TWIDDLES
def multi_precision_redc(T, N, B):
    N_prime = -pow(N, -1, B)
    p = int(ceil(log(N, B)))
    r = p
    R = B**r
    T_words = to_words(T, B, r + p)
    N_words = to_words(N, B, p)
    print(N, R, r, N_prime, N_words)
    # print(T_words)
    multiplies = 0
    for i in range(r):
        c = 0
        # This is the lo multiply
        m = (T_words[i] * N_prime) % B # N_prime = -1
        if N_prime != -1:
            multiplies += 1
        for j in range(p):
            # This is the hi multiply, then assert(c == 0?)
            x = T_words[i + j] + m * N_words[j] + c # N_words[0] = 1
            if N_words[j] != 1:
                multiplies += 1 # Karatsuba
            T_words[i + j] = x % B
            c = x // B
        # I wonder since I won't have overflow if this is needed
        # I suppose I can use 16 or 32 bit math?
        for j in range(p, r + p - i):
            x = T_words[i + j] + c
            T_words[i + j] = x % B
            c = x // B
        assert(c == 0)
        # print(T_words, from_words(T_words, B), from_words(T_words, B) % N)
    print(multiplies)
    s_words = T_words[r:r+p]
    s = from_words(s_words, B)
    if s > N:
        s -= N
    return s

def test_math(N, B):
    r = randint(0, (N-1)**2)
    correct = r % N
    mont_r = to_mont(r, B, N)
    ans = multi_precision_redc(mont_r, N, B)
    assert(ans == correct)

# Is there a simplification when B = 4096?  Yes
# What is the overhead of computing a 56 bit product over 32 bit integers? 6 multiplies for a reduce instead of 2

# I suppose 10 bits isn't any worse than 3 and you could do 16bit-float operations?
# It's 6 multiplies, not counting the initial product
# They also all need to be wide multiplies?  So they would be 32 bit.
# Since 32 bit is twice as fast that brings us back to 3 regular multiplies?
# I suppose you can force align it at 2^23 and then do bit operations to get high and lo parts?
# Then subtract away the extra?
# Maybe the last one can be a high multiply - but now you have to convert to half-floats anyway

# Could I do a mixed base with 2**12 2**32?  If the product was 44 bits.
# I suppose 2**15 doesn't make a nice N_prime?  It actually does, up until 2^16
# Then it becomes -65537 - which might be implementable with bitshifting?
# After 2^28 it then becomes nasty - but 1 word.
# This means that 14-16 also take two multiplies like 2**32 ?  But they are half of the width?


if __name__ == "__main__":
    # for i in range(1000):
    #     test_math(Q1, 4096)

    multi_precision_redc(0x0004000300020001, Q1, 2**16)
    multi_precision_redc(1, Q2, 2**16)

    multi_precision_redc(1, Q1, 4096)
    multi_precision_redc(1, Q1, 1024)
    # The product itself is another multiply
    multi_precision_redc(1, Q1, 2**32)
    multi_precision_redc(1, Q1*Q2, 2**32)

    for i in range(10, 32):
        print(i)
        multi_precision_redc(1, Q1, 2**i)
        multi_precision_redc(1, Q2, 2**i)
    # for i in range(1000):
    #     test_math(Q1, 2**16)