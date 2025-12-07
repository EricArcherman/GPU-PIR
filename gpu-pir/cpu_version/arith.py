from params import *
from structures import *

def mod_mult(a, b, mod):
    return (a * b) % mod

# May need to be applied to converted GSW texts
# Otherwise preapplied to constants
# Also apply to database
# Can do as mont multiply with R^2
def to_mont(a, mod):
    for i in range(len(a)):
        a[i] = (a[i] * 2**32) % mod
    return a

def from_mont(a, mod, mod_mont):
    b = []
    for i in range(len(a)):
        b.append(montgomery_mult(a[i], 1, mod, mod_mont, True))
    return b

# input (0, 14p), (0, p)
# output (0, 2p)
def montgomery_mult(a, b, P, J, correct=False):
    # w_prime is just w in montgomery form
    # u = t * w_prime
    u = ensure64bit(a * b)
    u_0 = ensure32bit(u)
    u_1 = u >> 32
    q = ensure32bit(u_0 * J)
    h = ensure64bit(q * P) >> 32
    y = u_1 - h
    if correct:
        return y + (y < 0) * P
    else:
        return y + P

def mod_mult_add(a, b, c, mod):
    return (c + a*b) % mod

# input (0, 2p) (0, 2p)
# output (0, 2p)
def mod_mult_add_poly(src1, src2, dst, mod, mod_mont, correct=True):
    for thread_idx in range(NUM_THREADS):
        for idx in range(0, POLY_LEN, NUM_THREADS):
            z = idx + thread_idx
            # dst[z] = mod_mult_add(src1[z], src2[z], dst[z], mod)
            out = dst[z] + montgomery_mult(src1[z], src2[z], mod, mod_mont, correct)
            # Can delay modulus more if you really want
            # Here we keep in (0, 2p)
            out -= 2*mod * (out >= 2*mod)
            if correct:
                out -= mod * (out >= mod)
            dst[z] = out

# input (0, 2p) (0, 2p)
# output (0, 2p)
def mod_mult_add_regs(r1: MultiRegisters, c1: PolyHalf, dst: MultiRegisters, mod, mont_mod, correct=True):
    for thread_idx in range(NUM_THREADS):
        for (j, idx) in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            z = idx + thread_idx
            out = dst.rs[thread_idx][j] + montgomery_mult(r1.rs[thread_idx][j], c1.data[z], mod, mont_mod, correct)
             # Can delay modulus more if you really want
            # Here we keep in (0, 2p)
            out -= 2*mod * (out >= 2*mod)
            if correct:
                out -= mod * (out >= mod)
            dst.rs[thread_idx][j] = out

def correct_one(s: Poly):
    for thread_idx in range(NUM_THREADS):
        for idx in range(0, POLY_LEN, NUM_THREADS):
            z = idx + thread_idx
            v = s.crt_0.data[z]
            v -= Q1 * (v >= Q1)
            s.crt_0.data[z] = v
    for thread_idx in range(NUM_THREADS):
        for idx in range(0, POLY_LEN, NUM_THREADS):
            z = idx + thread_idx
            v = s.crt_1.data[z]
            v -= Q2 * (v >= Q2)
            s.crt_1.data[z] = v



def inv_crt(a, b):
    a = montgomery_mult(a, CRT_INV_FOR1, Q1, Q1_MONT, True)
    b = montgomery_mult(b, CRT_INV_FOR2, Q2, Q2_MONT, True)
    s = a * Q2 + b * Q1
    if (s >= MODULUS):
        s -= MODULUS
    return s

def inv_crt_regs(r_Q1: MultiRegisters, r_Q2: MultiRegisters):
    for thread_idx in range(NUM_THREADS):
        for j in range(2 * NUM_PAIRS):
            c = inv_crt(r_Q1.rs[thread_idx][j], r_Q2.rs[thread_idx][j])
            r_Q1.rs[thread_idx][j] = c >> 32
            r_Q2.rs[thread_idx][j] = ensure32bit(c)

def inv_crt_poly(p: Poly):
    for thread_idx in range(NUM_THREADS):
        for idx in range(0, POLY_LEN, NUM_THREADS):
            z = idx + thread_idx
            i = inv_crt(p.crt_0.data[z], p.crt_1.data[z])
            p.crt_0.data[z] = i >> 32
            p.crt_1.data[z] = ensure32bit(i)

def add_regs(dst: MultiRegisters, src: MultiRegisters):
    for tid in range(NUM_THREADS):
        for i in range(NUM_REGS):
            dst.rs[tid][i] += src.rs[tid][i]

def reduce_regs(r: MultiRegisters, start, end, mod):
    while start > end:
        start = start // 2
        c = start * mod
        for tid in range(NUM_THREADS):
            for j in range(NUM_REGS):
                r.rs[tid][j] -= (c) * (r.rs[tid][j] >= (c))
        