import numpy as np
import random
q1 = 268369921
q2 = 249561089
qx = 536608769
qbits = 27
qbitx = 29

# I don't know what the barett with two shifts is
# Let's see what the bound with 1 is.
# def calc_barett(n):
#     for k in range(1, 64):
#         # Value of m, based on wikipedia
#         m = (2**k) // n
#         e = (1/n) - (m/(2**k))
#         theory_a = 1/e
#         if m > 0:
#             overflow_a = (2**64) // m
#             a = min(theory_a, overflow_a)
#             print(f"k={k}, m={m} ok for a={int(a)} ({np.log2(a)} bits)")

# 1 shift isn't good enough for product barrett, but is for single word input
# calc_barett(q1)
# calc_barett(q2)
# however x + 2^32 y mod q = x mod q + (2^32)mod q * y mod q
# So if 2^32 mod q = r is small, ry + x can be enough bits to fit in a barrett.
# can also use 2^34 or something 
# print(2**32 % q1)
# print(2**32 % q2)
# What does ntt_30bit do?
# it shifts by 29-2
# mu is 2^58 // q
# it shifts by 29+2
# then it multiplies again by q
# I think this is in the paper
# Not too critical unless computationally bound
# I just need it to be correct.
# So I might as well do the 2-shifts

# python has no concept of overflow, but I want to check correctness
def ensure64bit(x):
    return x & ((1 << 64) - 1)

def ensure32bit(x):
    return x & ((1 << 32) - 1)

def two_shift_barrett(a, q, mu, qbit):
    rx = a >> (qbit - 2)
    rx *= mu
    rx = ensure64bit(rx)
    rx >>= (qbit + 2)
    rx *= q
    rx = ensure64bit(rx)
    a -= rx

    a -= q * (a >= q)
    return a

def calc_mu(mod, qbit):
    return 2**(2*qbit) // mod

def check(val, mod, mu, qbit):
    correct = val % mod
    bar = two_shift_barrett(val, mod, mu, qbit)
    if correct != bar:
        print(f"{correct} {bar}")
    return correct == bar

def power_of_two_check(mod, mu, qbit):
    for i in range(28,64):
        print(f"2^{i}: {check(2**i, mod, mu, qbit)}")

# q1 is not well approximated by m/2^k and can't get to 56 bits
# q2 and qx can get to 61 bits

# mu1 = calc_mu(q1, qbits)
# mu2 = calc_mu(q2, qbits)
# mux = calc_mu(qx, qbitx)
# print(f"{q1} {mu1}")
# print(f"{q2} {mu2}")
# print(f"{qx} {mux}")
# power_of_two_check(q1, mu1, qbits)
# power_of_two_check(q2, mu2, qbits)
# power_of_two_check(qx, mux, qbitx)

# return mod^-1 mod 2^32
def calc_inv(mod):
    return pow(mod, -1, 2**32)

def montgomery(u, P, J):
    # w_prime is just w in montgomery form
    # u = t * w_prime
    u = ensure64bit(u)
    u_0 = ensure32bit(u)
    u_1 = u >> 32
    q = ensure32bit(u_0 * J)
    h = ensure64bit(q * P) >> 32
    y = u_1 - h
    return y + (y < 0) * P

def check_mont(val, mod, inv):
    rinv = pow(2**32, -1, mod)
    correct = (val * rinv) % mod
    # montgomery computes val/R mod n
    # where val is usually a product
    # I wonder if I do a scalar twiddle product if I can remove the division by R
    bar = montgomery(val, mod, inv)
    # It's the right modulus but it's too big when not equal
    if correct != bar % mod:
        print(f"{correct} {bar}")
    return correct == bar

def power_of_two_check_mont(mod):
    inv = calc_inv(mod)
    for i in range(28,64):
        print(f"2^{i}: {check_mont(2**i, mod, inv)}")

def random_check_mont(mod):
    inv = calc_inv(mod)
    minfail = 2**64
    for _ in range(10000):
        v = random.randint(minfail/2, minfail)
        if not (check_mont(v, mod, inv)):
            if v < minfail:
                minfail = v
    print(f"{mod}: {minfail} failed ({np.log2(minfail)} {minfail.bit_length()} bits)")
    print(f"{(minfail >> 32) / mod}")


power_of_two_check_mont(q1)
power_of_two_check_mont(q2)
random_check_mont(q1)
random_check_mont(q2)
# Both seem to indicate a failure between 59 and 60 bits
# too large of a range to iterate
# Seems to fail at 2^32 * p
# I bet a conditional correction on u_1 could allow you to go to
# 2^32 * 2p if you wanted
# It's clear that H is (2^32 * p) >> 32 is at most p so only u_1 causes the correction to fail
# Rather interestingly, this means it fails later with larger p?, but a smaller multiple of p
print(f"{q1} max ratio: {2**32 / q1}")
print(f"{q2} max ratio: {2**32 / q2}")
# Both are sufficient to get to 16p^2, so each in (0, 4p) or some combination thereof
# So you can remove some corrections and tolerate (0, 16p) input (with (0, p) twiddle) and output (0, 2p) always

def check_mont_hypothesis(mod):
    # should fail at 2^32 * mod
    inv = calc_inv(mod)
    print(f"{mod} {2**32 * mod} {check_mont(2**32 * mod, mod, inv)}")
    for i in range(2**32 * mod - 10000, 2**32 * mod):
        if not check_mont(i, mod, inv):
            print(f"{mod} error {i} failed!")

check_mont_hypothesis(q1)
check_mont_hypothesis(q2)

# For a modulus where 2^32 mod p is small (e.g 16 bits)
def my_mod(u, p):
    precomputed = (2**32) % p
    u = ensure64bit(u)
    u_0 = ensure32bit(u)
    u_1 = u >> 32
    # as 64 bit number addition and wide product again
    n = u_0 + u_1 * precomputed
    # now n has at most 32 + 16 bits, do other modulus reduction algorithm that works for that many bits

# If R mod n is small, then the conversion into montgomery can be done without overflow
# The binary decomosition input is (0, 1) so always can be
# The db is (0, 256) so can be if R mod n  is 20 bits, or perhaps with a few conditional corrections
# The binary decompositon for conversion is 7 or 8 bits so also (0, 256)

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

_, i1, i2 = egcd(q1, q2)
if (i1 < 0):
    i1 += q2
if (i2 < 0):
    i2 += q1

print(f"{i2}*{q1} + {i1} * {q2} = 1")

def inverse_crt(a, b):
    a = (a * i2) % q1
    b = (b * i1) % q2
    c = a * q2 + b * q1
    if (c > q1 * q2):
        c -= (q1 * q2)
    return c

def check_inverse(a, b):
    c = inverse_crt(a, b)
    if (c < 0) or (c >= q1 * q2):
        print("Out of range")
        return False
    if c % q1 != a:
        print("a")
        return False
    if c % q2 != b:
        print("b")
        return False
    return True

print(check_inverse(100, 250435))