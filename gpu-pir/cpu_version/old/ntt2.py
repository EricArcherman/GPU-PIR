import random
from sympy import ntt, intt
from structures import *
from params import *
from arith import *
from testing import *

def bitReverse(a, bit_length):
    res = 0
    for _ in range(bit_length):
        res <<= 1
        res = (a & 1) | res
        a >>= 1
    return res

def generate_twiddles(length, root, modulus):
    twiddles = [1 for _ in range(length)]
    for i in range(1, length):
        twiddles[i] = pow(root, bitReverse(i, LEN_LOG2), modulus)
    return twiddles

q1_twiddles = to_mont(generate_twiddles(POLY_LEN, Q1_ROOT, Q1), Q1)
q2_twiddles = to_mont(generate_twiddles(POLY_LEN, Q2_ROOT, Q2), Q2)
q1_inv_twiddles = to_mont(generate_twiddles(POLY_LEN, Q1_ROOT_INV, Q1), Q1)
q2_inv_twiddles = to_mont(generate_twiddles(POLY_LEN, Q2_ROOT_INV, Q2), Q2)
q1_corr = pow(POLY_LEN, -1, Q1)
q2_corr = pow(POLY_LEN, -1, Q2)

def const_load(idx, twiddles):
    # Load constants for first layers
    return twiddles[idx]

def butterfly(r, x_idx, y_idx, psi, modulus, inv_modulus):
    x = r[x_idx]
    y = r[y_idx]
    t = montgomery_mult(y, psi, modulus, inv_modulus, False)
    r[x_idx] = x + t
    r[y_idx] = x - t + 2*modulus

def align_idx(i, idx):
    return ((idx >> (LEN_LOG2 - 1 - i)) << (LEN_LOG2 - i)) ^ (idx & (((1 << (LEN_LOG2 - 1 - i)) - 1)))

def calc_idx(i, x_idx):
    return (1 << i) + (x_idx >> (LEN_LOG2 - i))

def calc_align_idx(i, idx):
    return (1 << i) + (idx >> (LEN_LOG2 - 1 - i))

def radix_ntt(r : List[Registers], scaling, twiddles, mod, mont_mod):
    # for NUM_RADIX = 4
    #8 # 1
    #4 # 2
    #2 # 4
    #1 # 8
    for i in range(NUM_RADIX):
        a = (1 << (NUM_RADIX - i - 1))
        b = (1 << i)
        sm = [0 for _ in range(POLY_LEN)]
        sm_input = [0 for _ in range(POLY_LEN)]
        for thread_idx in range(NUM_THREADS):
            s = 0
            si = 0
            for k in range(b):
                for j in range(a):
                    if scaling > 0:
                        # same for all threads in a warp
                        warp = thread_idx >> 5
                        # Can be removed for specific numbers of threads
                        warp_idx = thread_idx & (WARP_SIZE - 1)
                        idx = warp * (WARP_SIZE * NUM_PAIRS) + warp_idx
                        idx += (si + j) * WARP_SIZE
                        twiddle_idx = calc_align_idx(i + scaling, idx) 
                    else:
                        # optimal case - same for all threads
                        idx = ((si + j) << (LEN_LOG2 - scaling - NUM_RADIX)) + thread_idx
                        twiddle_idx = k + b
                    twiddle = const_load(twiddle_idx, twiddles)
                    x, y = r[thread_idx][s+ j], r[thread_idx][s + j + a]
                    butterfly(r[thread_idx], s+j, s+j+a, twiddle, mod, mont_mod)
                    x_idx, y_idx = align(i + scaling, idx)
                    # print(f"{i + scaling} {x_idx} {y_idx} {twiddle_idx} {twiddles[twiddle_idx]} {x % mod} {y % mod} {r[thread_idx][s+j] % mod} {r[thread_idx][s+j+a] % mod}")
                    sm[x_idx], sm[y_idx] = r[thread_idx][s+j] % mod, r[thread_idx][s+j+a] % mod
                    sm_input[x_idx], sm_input[y_idx] = x % mod, y % mod
                s += 2*a
                si += a
        # print(f"{i + scaling} {test_hash(sm)} ({test_hash(sm_input)})")
def write_to_sm(r: Registers, sm):
    for thread_idx in range(NUM_THREADS):
        for i in range(2*NUM_PAIRS):
            sm[i*(POLY_LEN // NUM_PAIRS // 2) + thread_idx] = r[thread_idx][i]

def read_from_sm(r: Registers, sm):
    for thread_idx in range(NUM_THREADS):
        warp = thread_idx >> 5
        warp_idx = thread_idx & (WARP_SIZE - 1)
        idx = warp * (WARP_SIZE * NUM_PAIRS) + warp_idx
        for j in range(NUM_PAIRS):
            x_idx, y_idx = align(NUM_RADIX, idx)
            r[thread_idx][j] = sm[x_idx]
            r[thread_idx][j+NUM_PAIRS] = sm[y_idx]
            idx += WARP_SIZE

def shuffle(local, swap, idx_x, idx_y, tid, mask):
    dir = tid & mask
    to_idx = (mask) ^ tid
    if dir:
        # send x keep y
        send = local[tid][idx_x]
        swap[tid][idx_y] = local[tid][idx_y]
    else:
        # send y keep x
        send = local[tid][idx_y]
        swap[tid][idx_x] = local[tid][idx_x]
    # The partner has the negation of dir
    # writes the sent value to what they sent
    if not dir:
        swap[to_idx][idx_x] = send
    else:
        swap[to_idx][idx_y] = send

def align(i, idx):
    p = 1 << (LEN_LOG2 - 1 - i)
    x = ((idx >> (LEN_LOG2 - 1 - i)) << (LEN_LOG2 - i)) ^ (idx & (((1 << (LEN_LOG2 - i - 1)) - 1)))
    y = x ^ p
    return x, y

def shuffle_ntt(r: List[Registers], start, end, twiddles, mod, mont_mod):
    for i in range(start, end):
        if i == 8:
            mod_correct(r, mod)
        mask = 1 << (LEN_LOG2 - i - 1)
        swap = [Registers() for _ in range(NUM_THREADS)]
        for thread_idx in range(NUM_THREADS):
            for j in range(NUM_PAIRS):
                shuffle(r, swap, 2*j, 2*j+1, thread_idx, mask) 
        r = swap
        sm = [0 for _ in range(POLY_LEN)]
        for thread_idx in range(NUM_THREADS):
            warp = thread_idx >> 5
            warp_idx = thread_idx & (WARP_SIZE - 1)
            idx = warp * (WARP_SIZE * NUM_PAIRS) + warp_idx
            for j in range(NUM_PAIRS):
                twiddle_idx = calc_align_idx(i, idx)
                x, y = r[thread_idx][2*j], r[thread_idx][2*j+1]
                butterfly(r[thread_idx], 2*j, 2*j+1, twiddles[twiddle_idx], mod, mont_mod)
                x_idx, y_idx = align(i, idx)
                sm[x_idx] = r[thread_idx][2 * j]
                sm[y_idx] = r[thread_idx][2 * j + 1]
                # print(f"{i} {x_idx} {y_idx} {twiddle_idx} {twiddles[twiddle_idx]} {x % mod} {y % mod} {r[thread_idx][2*j] % mod} {r[thread_idx][2*j+1] % mod}")
                idx += WARP_SIZE
        sm = [x % mod for x in sm]
        # print(f"{i} {test_hash(sm)}")
    return r

# Can apply this modulus correction anywhere between (5 and 8) of the 11 iterations
def mod_correct(r: List[Registers], mod):
    for thread_idx in range(NUM_THREADS):
        for i in range(2*NUM_PAIRS):
            r[thread_idx][i] -= (8 * mod) * (r[thread_idx][i] >= (8 * mod))

def ntt_x(r:List[Registers], twiddles, mod, mont_mod):
    r = shuffle_ntt(r, 0, LEN_LOG2 - (5+2*NUM_RADIX), twiddles, mod, mont_mod)
    radix_ntt(r, LEN_LOG2 - (5+2*NUM_RADIX), twiddles, mod, mont_mod)
    sm = [0 for _ in range(POLY_LEN)]
    write_to_sm(r, sm)
    # Sync threads
    read_from_sm(r, sm)
    radix_ntt(r, LEN_LOG2 - (5+NUM_RADIX), twiddles, mod, mont_mod)
    return shuffle_ntt(r, LEN_LOG2 - 5, LEN_LOG2, twiddles, mod, mont_mod)

def gpu_ntt_x(shared, modulus, inv_modulus, twiddles, correct=True):
    r = [Registers() for _ in range(NUM_THREADS)]
    for thread_idx in range(NUM_THREADS):
         for l, i in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            r[thread_idx][l] = shared[i + thread_idx]
    r = ntt_x(r, twiddles, modulus, inv_modulus)
    for thread_idx in range(NUM_THREADS):
        warp = thread_idx >> 5
        warp_idx = thread_idx & (WARP_SIZE - 1)
        idx = warp * (WARP_SIZE * NUM_PAIRS) + warp_idx
        for j in range(NUM_PAIRS):
            x_idx, y_idx = align(LEN_LOG2 - 1, idx)
            shared[x_idx] = r[thread_idx][2 * j]
            shared[y_idx] = r[thread_idx][2 * j + 1]
            idx += WARP_SIZE
    if correct:
        for i in range(len(shared)):
            if shared[i] >= 2**32:
                print(f"{i} overflow!")
            shared[i] = shared[i] % modulus
    return shared

def ntt_regs(r : MultiRegisters, mod, mont_mod, twiddles, correct=True):
    r.rs = ntt_x(r.rs, twiddles, mod, mont_mod)
    if correct:
        for thread_idx in range(NUM_THREADS):
            for i in range(2*NUM_PAIRS):
                if r.rs[thread_idx][i] >= 2**32:
                    print(f"{thread_idx} {i} overflow!")
    return r

def intt_butterfly(r, x_idx, y_idx, psi, modulus, inv_modulus):
    x = r[x_idx]
    y = r[y_idx]
    x_prime = x + y
    x_prime -= modulus * (x_prime >= modulus)
    t = x + modulus - y
    y_prime = montgomery_mult(psi, t, modulus, inv_modulus, False)
    y_prime -= modulus * (y_prime >= modulus)
    r[x_idx] = x_prime
    r[y_idx] = y_prime

def shuffle_intt(r: List[Registers], start, end, twiddles, mod, mont_mod):
    for i in range(start, end).__reversed__():
        mask = 1 << (LEN_LOG2 - i - 1)
        sm = [0 for _ in range(POLY_LEN)]
        for thread_idx in range(NUM_THREADS):
            warp = thread_idx >> 5
            warp_idx = thread_idx & (WARP_SIZE - 1)
            idx = warp * (WARP_SIZE * NUM_PAIRS) + warp_idx
            for j in range(NUM_PAIRS):
                twiddle_idx = calc_align_idx(i, idx)
                x, y = r[thread_idx][2*j], r[thread_idx][2*j+1]
                intt_butterfly(r[thread_idx], 2*j, 2*j+1, twiddles[twiddle_idx], mod, mont_mod)
                x_idx, y_idx = align(i, idx)
                sm[x_idx] = r[thread_idx][2 * j]
                sm[y_idx] = r[thread_idx][2 * j + 1]
                # print(f"{i} {x_idx} {y_idx} {twiddle_idx} {twiddles[twiddle_idx]} {x % mod} {y % mod} {r[thread_idx][2*j] % mod} {r[thread_idx][2*j+1] % mod}")
                idx += WARP_SIZE
        swap = [Registers() for _ in range(NUM_THREADS)]
        for thread_idx in range(NUM_THREADS):
            for j in range(NUM_PAIRS):
                shuffle(r, swap, 2*j, 2*j+1, thread_idx, mask) 
        r = swap
        sm = [x % mod for x in sm]
        # print(f"{i} {test_hash(sm)}")
    return r

def radix_intt(r : List[Registers], scaling, twiddles, mod, mont_mod):
    for i in range(0, NUM_RADIX).__reversed__():
        a = (1 << (NUM_RADIX - i - 1))
        b = (1 << i)
        sm = [0 for _ in range(POLY_LEN)]
        sm_input = [0 for _ in range(POLY_LEN)]
        for thread_idx in range(NUM_THREADS):
            s = 0
            si = 0
            for k in range(b):
                for j in range(a):
                    if scaling > 0:
                        # same for all threads in a warp
                        warp = thread_idx >> 5
                        # Can be removed for specific numbers of threads
                        warp_idx = thread_idx & (WARP_SIZE - 1)
                        idx = warp * (WARP_SIZE * NUM_PAIRS) + warp_idx
                        idx += (si + j) * WARP_SIZE
                        twiddle_idx = calc_align_idx(i + scaling, idx) 
                    else:
                        # optimal case - same for all threads
                        idx = ((si + j) << (LEN_LOG2 - scaling - NUM_RADIX)) + thread_idx
                        twiddle_idx = k + b
                    twiddle = const_load(twiddle_idx, twiddles)
                    x, y = r[thread_idx][s+ j], r[thread_idx][s + j + a]
                    intt_butterfly(r[thread_idx], s+j, s+j+a, twiddle, mod, mont_mod)
                    x_idx, y_idx = align(i + scaling, idx)
                    # print(f"{i + scaling} {x_idx} {y_idx} {twiddle_idx} {twiddles[twiddle_idx]} {x % mod} {y % mod} {r[thread_idx][s+j] % mod} {r[thread_idx][s+j+a] % mod}")
                    sm[x_idx], sm[y_idx] = r[thread_idx][s+j] % mod, r[thread_idx][s+j+a] % mod
                    sm_input[x_idx], sm_input[y_idx] = x % mod, y % mod
                s += 2*a
                si += a
        # print(f"{i + scaling} {test_hash(sm)} ({test_hash(sm_input)})")

def iread_sm(r: Registers, sm):
    for thread_idx in range(NUM_THREADS):
        for i in range(2*NUM_PAIRS):
            r[thread_idx][i] = sm[i*(POLY_LEN // NUM_PAIRS // 2) + thread_idx]

def iwrite_sm(r: Registers, sm):
    for thread_idx in range(NUM_THREADS):
        warp = thread_idx >> 5
        warp_idx = thread_idx & (WARP_SIZE - 1)
        idx = warp * (WARP_SIZE * NUM_PAIRS) + warp_idx
        for j in range(NUM_PAIRS):
            x_idx, y_idx = align(NUM_RADIX, idx)
            sm[x_idx] = r[thread_idx][j] 
            sm[y_idx] = r[thread_idx][j+NUM_PAIRS] 
            idx += WARP_SIZE

def intt_x(r:List[Registers], twiddles, mod, mont_mod):
    r = shuffle_intt(r, LEN_LOG2 - 5, LEN_LOG2, twiddles, mod, mont_mod)
    radix_intt(r, LEN_LOG2 - (5+NUM_RADIX), twiddles, mod, mont_mod)
    sm = [0 for _ in range(POLY_LEN)]
    iwrite_sm(r, sm)
    # Sync threads
    iread_sm(r, sm)
    radix_intt(r, LEN_LOG2 - (5+2*NUM_RADIX), twiddles, mod, mont_mod)
    return shuffle_intt(r, 0, LEN_LOG2 - (5+2*NUM_RADIX), twiddles, mod, mont_mod)

def gpu_intt_x(shared, modulus, inv_modulus, twiddles, correct=True):
    r = [Registers() for _ in range(NUM_THREADS)]
    for thread_idx in range(NUM_THREADS):
        warp = thread_idx >> 5
        warp_idx = thread_idx & (WARP_SIZE - 1)
        idx = warp * (WARP_SIZE * NUM_PAIRS) + warp_idx
        for j in range(NUM_PAIRS):
            x_idx, y_idx = align(LEN_LOG2 - 1, idx)
            r[thread_idx][2 * j] = shared[x_idx]
            r[thread_idx][2 * j + 1] = shared[y_idx]
            idx += WARP_SIZE
    r = intt_x(r, twiddles, modulus, inv_modulus)
    for thread_idx in range(NUM_THREADS):
         for l, i in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            shared[i + thread_idx] = r[thread_idx][l]
    if correct:
        for i in range(len(shared)):
            if shared[i] >= 2**32:
                print(f"{i} overflow!")
            shared[i] = shared[i] % modulus
        scale_factor = q1_corr if modulus == Q1 else q2_corr
        shared = [(a * scale_factor) % modulus  for a in shared]
    return shared


def intt_regs(r : MultiRegisters, mod, mont_mod, twiddles, correct=True):
    r.rs = intt_x(r.rs, twiddles, mod, mont_mod)
    if correct:
        scale_factor = q1_corr if mod == Q1 else q2_corr
        for thread_idx in range(NUM_THREADS):
            for i in range(2*NUM_PAIRS):
                if r.rs[thread_idx][i] >= 2**32:
                    print(f"{thread_idx} {i} overflow!")
                r.rs[thread_idx][i] = (r.rs[thread_idx][i] * scale_factor) % mod
    return r

def test_nttx():
    poly1 = [0 for _ in range(POLY_LEN)]
    poly2 = [0 for _ in range(POLY_LEN)]
    poly1[0] = Q1 - 1
    poly1[1] = 1
    for i in range(5):
        poly2[i] = 1
    p1n1 = ntt(poly1, Q1)
    p2n1 = ntt(poly2, Q1)
    p3n1 = [a * p2n1[idx] % Q1 for idx, a in enumerate(p1n1)]
    product1 = intt(p3n1, Q1)
    print(product1[:10])

    p1n2 = gpu_ntt_x(poly1, Q1, Q1_MONT, q1_twiddles)
    p2n2 = gpu_ntt_x(poly2,  Q1, Q1_MONT, q1_twiddles)
    p3n2 = [a * p2n2[idx] % Q1 for idx, a in enumerate(p1n2)]
    product2 = gpu_intt_x(p3n2,  Q1, Q1_MONT, q1_inv_twiddles)
    print(product2[:10])
    print(gpu_intt_x(p1n2, Q1, Q1_MONT, q1_inv_twiddles)[0:10])
    print(gpu_intt_x(p2n2, Q1, Q1_MONT, q1_inv_twiddles)[0:10])

def test_ntt_equivalent():
    raw = load_polys("raw_poly.dat", False, 0, False)[0]
    crt0 = PolyHalf()
    crt1 = PolyHalf()
    for i in range(POLY_LEN):
        v = (raw.crt_0.data[i] << 32) + raw.crt_1.data[i]
        crt0.data[i] = v % Q1
        crt1.data[i] = v % Q2
    x = Poly(crt0, crt1)
    gpu_ntt_x(x.crt_0.data, Q1, Q1_MONT, q1_twiddles)
    gpu_ntt_x(x.crt_1.data, Q2, Q2_MONT, q2_twiddles)
    ntt_form = load_polys("ntt_poly.dat", True, 0, False)[0]
    for i in range(POLY_LEN):
        if x.crt_0.data[i] % Q1 != ntt_form.crt_0.data[i] % Q1:
            print(f"crt0 {i} {x.crt_0.data[i]} {ntt_form.crt_0.data[i]}")
        if x.crt_1.data[i] % Q2 != ntt_form.crt_1.data[i] % Q2:
            print(f"crt1 {i} {x.crt_1.data[i]} {ntt_form.crt_1.data[i]}")
    gpu_intt_x(ntt_form.crt_0.data, Q1, Q1_MONT, q1_inv_twiddles, False)
    gpu_intt_x(ntt_form.crt_1.data, Q2, Q2_MONT, q2_inv_twiddles, False)
    inv_crt_poly(ntt_form)
    for i in range(POLY_LEN):
        v = (raw.crt_0.data[i] << 32) + raw.crt_1.data[i]
        v2 = (ntt_form.crt_0.data[i] << 32) + ntt_form.crt_1.data[i]
        if v % MODULUS != v2 % MODULUS: 
            print(f"raw {i} {v} {v2}")

def test_intt():
    poly1 = [0 for _ in range(POLY_LEN)]
    poly2 = [0 for _ in range(POLY_LEN)]
    poly1[0] = Q1 - 1
    poly1[1] = 1
    for i in range(5):
        poly2[i] = 1
    p1n1 = ntt(poly1, Q1)
    p2n1 = ntt(poly2, Q1)
    p3n1 = [a * p2n1[idx] % Q1 for idx, a in enumerate(p1n1)]
    product1 = intt(p3n1, Q1)
    print(product1[:10])

    p1n2 = gpu_ntt_x(poly1, Q1, Q1_MONT, q1_twiddles)
    p2n2 = gpu_ntt_x(poly2,  Q1, Q1_MONT, q1_twiddles)
    p3n2 = [a * p2n2[idx] % Q1 for idx, a in enumerate(p1n2)]
    product2 = gpu_intt_x(p3n2,  Q1, Q1_MONT, q1_inv_twiddles)
    print(product2[:10])
    print(gpu_intt_x(p1n2, Q1, Q1_MONT, q1_inv_twiddles)[0:10])
    print(gpu_intt_x(p2n2, Q1, Q1_MONT, q1_inv_twiddles)[0:10])

if __name__ == "__main__":
    test_nttx()
    test_ntt_equivalent()
    