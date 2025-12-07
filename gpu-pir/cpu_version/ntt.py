from params import *
from structures import *
from arith import *
from sympy import ntt, intt
from testing import *
from copy import deepcopy

# 256 thread hardcoded
# use align to put the twiddles in well-strided locations

# just do radix-shuffle-radix for simplicity
# inverse NTT should be just in reverse ordering
# make work on GPU
# perhaps register struct, but not if continues to introduce branches


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
        twiddles[i] = pow(root, bitReverse(i, 11), modulus)
    return twiddles

def restride_twiddles(og):
    n = [1 for _ in range(len(og))]
    for i in range(512):
        n[i] = og[i]
    for tid in range(256):
        twiddle_idx = 512 + 2 * tid
        twiddle_base = 512 + tid
        n[twiddle_base] = og[twiddle_idx]
        n[twiddle_base + 256] = og[twiddle_idx + 1]
    for tid in range(256):
        twiddle_idx = 1024 + 4 * tid
        twiddle_base = 1024 + tid
        n[twiddle_base] = og[twiddle_idx]
        n[twiddle_base + 256] = og[twiddle_idx + 1]
        n[twiddle_base + 512] = og[twiddle_idx + 2]
        n[twiddle_base + (3*256)] = og[twiddle_idx+3]
    return n

q1_twiddles = restride_twiddles(to_mont(generate_twiddles(POLY_LEN, Q1_ROOT, Q1), Q1))
q2_twiddles = restride_twiddles(to_mont(generate_twiddles(POLY_LEN, Q2_ROOT, Q2), Q2))
q1_inv_twiddles = restride_twiddles(to_mont(generate_twiddles(POLY_LEN, Q1_ROOT_INV, Q1), Q1))
q2_inv_twiddles = restride_twiddles(to_mont(generate_twiddles(POLY_LEN, Q2_ROOT_INV, Q2), Q2))
q1_corr = pow(POLY_LEN, -1, Q1)
q2_corr = pow(POLY_LEN, -1, Q2)

def align_idx(i, idx):
    return ((idx >> (LEN_LOG2 - 1 - i)) << (LEN_LOG2 - i)) ^ (idx & (((1 << (LEN_LOG2 - 1 - i)) - 1)))

def calc_idx(i, x_idx):
    return (1 << i) + (x_idx >> (LEN_LOG2 - i))

def calc_align_idx(i, idx):
    return (1 << i) + (idx >> (LEN_LOG2 - 1 - i))


def butterfly(r, x_idx, y_idx, psi, modulus, inv_modulus):
    x = r[x_idx]
    y = r[y_idx]
    t = montgomery_mult(y, psi, modulus, inv_modulus, False)
    r[x_idx] = x + t
    r[y_idx] = x - t + 2*modulus

def butterfly_swap(local, swap, idx_x, idx_y, tid, mask):
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

# input (0, p)
# output (0, 14p)
def ntt_256(r,modulus,inv_modulus,twiddles):
    # test_hash_reduce(r, "input")
    # Stride 1024
    for tid in range(256):
        butterfly(r[tid], 0, 4, twiddles[1], modulus, inv_modulus)
        butterfly(r[tid], 1, 5, twiddles[1], modulus, inv_modulus)
        butterfly(r[tid], 2, 6, twiddles[1], modulus, inv_modulus)
        butterfly(r[tid], 3, 7, twiddles[1], modulus, inv_modulus)
    # Stride 512
    for tid in range(256):
        butterfly(r[tid], 0, 2, twiddles[2], modulus, inv_modulus)
        butterfly(r[tid], 1, 3, twiddles[2], modulus, inv_modulus)
        butterfly(r[tid], 4, 6, twiddles[3], modulus, inv_modulus)
        butterfly(r[tid], 5, 7, twiddles[3], modulus, inv_modulus)
    # Stride 256
    for tid in range(256):
        butterfly(r[tid], 0, 1, twiddles[4], modulus, inv_modulus)
        butterfly(r[tid], 2, 3, twiddles[5], modulus, inv_modulus)
        butterfly(r[tid], 4, 5, twiddles[6], modulus, inv_modulus)
        butterfly(r[tid], 6, 7, twiddles[7], modulus, inv_modulus)
    # test_hash_reduce(r, "before shared")
    # Here we use shared memory to swap values between warps
    shared = [0 for _ in range(2048)]
    for tid in range(256):
        for i in range(8):
            shared[i*256 + tid] = r[tid][i]
    # __syncthreads()
    # Read 2 sets of 4 consecutive values, spaced by 128
    # Each warp has 256 consecutive values
    
    for tid in range(256):
        warp = tid >> 5
        l = tid & 31
        for i in range(4):
            x = 4 * l + 256 * warp + i
            r[tid][i] = shared[x]
            r[tid][i + 4] = shared[x + 128]
    # test_hash_reduce(r, "after shared")
    for i in range(3,LEN_LOG2-3):
        for tid in range(256):
            # I wonder if you could compute this with one multiply from the previous value
            # Multiply by a if the bit now used is 0, and by b if the bit now used is 1
            twiddle_idx = (1 << i) + (tid >> (8 - i))
            butterfly(r[tid], 0, 4, twiddles[twiddle_idx], modulus, inv_modulus)
            butterfly(r[tid], 1, 5, twiddles[twiddle_idx], modulus, inv_modulus)
            butterfly(r[tid], 2, 6, twiddles[twiddle_idx], modulus, inv_modulus)
            butterfly(r[tid], 3, 7, twiddles[twiddle_idx], modulus, inv_modulus)

        # test_hash_reduce(r, "butterfly " + str(i))

        swapped = [[0 for _ in range(8)] for _ in range(256)]
        shift = 1 << (7 - i)
        for tid in range(256):
            butterfly_swap(r, swapped, 0, 4, tid, shift)
            butterfly_swap(r, swapped, 1, 5, tid, shift)
            butterfly_swap(r, swapped, 2, 6, tid, shift)
            butterfly_swap(r, swapped, 3, 7, tid, shift)
        r = swapped

        # test_hash_reduce(r, "shuffle " + str(i))
    # Can apply this modulus correction anywhere between (5 and 8) of the 11 iterations
    for tid in range(256):
        # can be just the x-values (iterate through 4)
        for i in range(8):
            r[tid][i] -= (8 * modulus) * (r[tid][i] >= (8 * modulus))


    # Stride 4
    for tid in range(256):
        twiddle_idx = 256 + 1 * tid
        butterfly(r[tid], 0, 4, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 1, 5, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 2, 6, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 3, 7, twiddles[twiddle_idx], modulus, inv_modulus)
    
    # test_hash_reduce(r, "after 4");
    # Stride 2
    for tid in range(256):
        # twiddle_idx = 512 + 2 * tid
        # butterfly(r[tid], 0, 2, twiddles[twiddle_idx], modulus, inv_modulus)
        # butterfly(r[tid], 1, 3, twiddles[twiddle_idx], modulus, inv_modulus)
        # butterfly(r[tid], 4, 6, twiddles[twiddle_idx+1], modulus, inv_modulus)
        # butterfly(r[tid], 5, 7, twiddles[twiddle_idx+1], modulus, inv_modulus)

        twiddle_base = 512 + tid
        butterfly(r[tid], 0, 2, twiddles[twiddle_base], modulus, inv_modulus)
        butterfly(r[tid], 1, 3, twiddles[twiddle_base], modulus, inv_modulus)
        butterfly(r[tid], 4, 6, twiddles[twiddle_base+256], modulus, inv_modulus)
        butterfly(r[tid], 5, 7, twiddles[twiddle_base+256], modulus, inv_modulus)
   
    # test_hash_reduce(r, "after 2");
    # Stride 1
    for tid in range(256):
        # twiddle_idx = 1024 + 4 * tid
        # butterfly(r[tid], 0, 1, twiddles[twiddle_idx], modulus, inv_modulus)
        # butterfly(r[tid], 2, 3, twiddles[twiddle_idx+1], modulus, inv_modulus)
        # butterfly(r[tid], 4, 5, twiddles[twiddle_idx+2], modulus, inv_modulus)
        # butterfly(r[tid], 6, 7, twiddles[twiddle_idx+3], modulus, inv_modulus)

        twiddle_base = 1024 + tid
        butterfly(r[tid], 0, 1, twiddles[twiddle_base], modulus, inv_modulus)
        butterfly(r[tid], 2, 3, twiddles[twiddle_base+256], modulus, inv_modulus)
        butterfly(r[tid], 4, 5, twiddles[twiddle_base+512], modulus, inv_modulus)
        butterfly(r[tid], 6, 7, twiddles[twiddle_base+(3*256)], modulus, inv_modulus)
    
    # test_hash_reduce(r, "after ntt_256")
    return r

def gpu_ntt_256(shared, modulus, inv_modulus, twiddles, correct=True):
    r = [[0 for _ in range(8)] for _ in range(256)]
    # copy global to local - likely input will already be in registers from bitshift operation
    for thread_idx in range(0, 256):
        for l, i in enumerate(range(0, 2048, 256)):
            r[thread_idx][l] = shared[i + thread_idx]
    r = ntt_256(r, modulus, inv_modulus, twiddles)
    for thread_idx in range(0, 256):
        for i in range(8):
            shared[thread_idx*8 + i] = r[thread_idx][i]
    if correct:
        for i in range(len(shared)):
            if shared[i] >= 2**32:
                print(f"{i} overflow!")
            shared[i] = shared[i] % modulus
    return shared

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

# input (0,p)
# output (0, p)
def intt_256(r,modulus,inv_modulus,twiddles):
    # test_hash_reduce(r, "intt_256 input")
    # Stride 1
    for tid in range(256):
        # twiddle_idx = 1024 + 4 * tid
        # intt_butterfly(r[tid], 0, 1, twiddles[twiddle_idx], modulus, inv_modulus)
        # intt_butterfly(r[tid], 2, 3, twiddles[twiddle_idx+1], modulus, inv_modulus)
        # intt_butterfly(r[tid], 4, 5, twiddles[twiddle_idx+2], modulus, inv_modulus)
        # intt_butterfly(r[tid], 6, 7, twiddles[twiddle_idx+3], modulus, inv_modulus)
        twiddle_base = 1024 + tid
        intt_butterfly(r[tid], 0, 1, twiddles[twiddle_base], modulus, inv_modulus)
        intt_butterfly(r[tid], 2, 3, twiddles[twiddle_base+256], modulus, inv_modulus)
        intt_butterfly(r[tid], 4, 5, twiddles[twiddle_base+512], modulus, inv_modulus)
        intt_butterfly(r[tid], 6, 7, twiddles[twiddle_base+(3*256)], modulus, inv_modulus)
    # Stride 2
    for tid in range(256):
        # twiddle_idx = 512 + 2 * tid
        # intt_butterfly(r[tid], 0, 2, twiddles[twiddle_idx], modulus, inv_modulus)
        # intt_butterfly(r[tid], 1, 3, twiddles[twiddle_idx], modulus, inv_modulus)
        # intt_butterfly(r[tid], 4, 6, twiddles[twiddle_idx+1], modulus, inv_modulus)
        # intt_butterfly(r[tid], 5, 7, twiddles[twiddle_idx+1], modulus, inv_modulus)

        twiddle_base = 512 + tid
        intt_butterfly(r[tid], 0, 2, twiddles[twiddle_base], modulus, inv_modulus)
        intt_butterfly(r[tid], 1, 3, twiddles[twiddle_base], modulus, inv_modulus)
        intt_butterfly(r[tid], 4, 6, twiddles[twiddle_base+256], modulus, inv_modulus)
        intt_butterfly(r[tid], 5, 7, twiddles[twiddle_base+256], modulus, inv_modulus)
    # Stride 4
    for tid in range(256):
        twiddle_idx = 256 + 1 * tid
        intt_butterfly(r[tid], 0, 4, twiddles[twiddle_idx], modulus, inv_modulus)
        intt_butterfly(r[tid], 1, 5, twiddles[twiddle_idx], modulus, inv_modulus)
        intt_butterfly(r[tid], 2, 6, twiddles[twiddle_idx], modulus, inv_modulus)
        intt_butterfly(r[tid], 3, 7, twiddles[twiddle_idx], modulus, inv_modulus)
    # test_hash_reduce(r, "after radix")
    for i in range(3,LEN_LOG2-3).__reversed__():
        swapped = [[0 for _ in range(8)] for _ in range(256)]
        shift = 1 << (7 - i)
        for tid in range(256):
            butterfly_swap(r, swapped, 0, 4, tid, shift)
            butterfly_swap(r, swapped, 1, 5, tid, shift)
            butterfly_swap(r, swapped, 2, 6, tid, shift)
            butterfly_swap(r, swapped, 3, 7, tid, shift)
        r = swapped

        for tid in range(256):
            twiddle_idx = (1 << i) + (tid >> (8 - i))
            intt_butterfly(r[tid], 0, 4, twiddles[twiddle_idx], modulus, inv_modulus)
            intt_butterfly(r[tid], 1, 5, twiddles[twiddle_idx], modulus, inv_modulus)
            intt_butterfly(r[tid], 2, 6, twiddles[twiddle_idx], modulus, inv_modulus)
            intt_butterfly(r[tid], 3, 7, twiddles[twiddle_idx], modulus, inv_modulus)
    # test_hash_reduce(r, "after shuffles")
    # write 2 sets of 4 consecutive values, spaced by 128
    # Each warp has 256 consecutive values
    shared = [0 for _ in range(2048)]
    for tid in range(256):
        warp = tid >> 5
        l = tid & 31
        for i in range(4):
            x = 4 * l + 256 * warp + i
            shared[x] = r[tid][i]
            shared[x + 128] = r[tid][i + 4]
    
    for tid in range(256):
        for i in range(8):
            r[tid][i] = shared[i*256 + tid]
    # test_hash_reduce(r, "after shared")
    # Stride 256
    for tid in range(256):
        intt_butterfly(r[tid], 0, 1, twiddles[4], modulus, inv_modulus)
        intt_butterfly(r[tid], 2, 3, twiddles[5], modulus, inv_modulus)
        intt_butterfly(r[tid], 4, 5, twiddles[6], modulus, inv_modulus)
        intt_butterfly(r[tid], 6, 7, twiddles[7], modulus, inv_modulus)
    # Stride 512
    for tid in range(256):
        intt_butterfly(r[tid], 0, 2, twiddles[2], modulus, inv_modulus)
        intt_butterfly(r[tid], 1, 3, twiddles[2], modulus, inv_modulus)
        intt_butterfly(r[tid], 4, 6, twiddles[3], modulus, inv_modulus)
        intt_butterfly(r[tid], 5, 7, twiddles[3], modulus, inv_modulus)
    # Stride 1024
    for tid in range(256):
        intt_butterfly(r[tid], 0, 4, twiddles[1], modulus, inv_modulus)
        intt_butterfly(r[tid], 1, 5, twiddles[1], modulus, inv_modulus)
        intt_butterfly(r[tid], 2, 6, twiddles[1], modulus, inv_modulus)
        intt_butterfly(r[tid], 3, 7, twiddles[1], modulus, inv_modulus)
    # test_hash_reduce(r, "after intt_256")
    return r

def gpu_intt_256(shared, modulus, inv_modulus, twiddles, correct=True):
    r = [[0 for _ in range(8)] for _ in range(256)]
    for thread_idx in range(0, 256):
        for i in range(8):
            r[thread_idx][i] = shared[thread_idx*8 + i]
    r = intt_256(r, modulus, inv_modulus, twiddles)
    
    for thread_idx in range(0, 256):
        for l, i in enumerate(range(0, 2048, 256)):
            shared[i + thread_idx] = r[thread_idx][l] 
    if correct:
        for i in range(len(shared)):
            if shared[i] >= 2**32:
                print(f"{i} overflow!")
            shared[i] = shared[i] % modulus
        scale_factor = q1_corr if modulus == Q1 else q2_corr
        shared = [(a * scale_factor) % modulus  for a in shared]
    return shared

def ntt_regs(r : MultiRegisters, mod, mont_mod, twiddles, correct=True):
    r.rs = ntt_256(r.rs, mod, mont_mod, twiddles)
    if correct:
        for thread_idx in range(NUM_THREADS):
            for i in range(2*NUM_PAIRS):
                if r.rs[thread_idx][i] >= 2**32:
                    print(f"{thread_idx} {i} overflow!")
    return r

def intt_regs(r : MultiRegisters, mod, mont_mod, twiddles, correct=True):
    r.rs = intt_256(r.rs, mod, mont_mod, twiddles)
    if correct:
        scale_factor = q1_corr if mod == Q1 else q2_corr
        for thread_idx in range(NUM_THREADS):
            for i in range(2*NUM_PAIRS):
                if r.rs[thread_idx][i] >= 2**32:
                    print(f"{thread_idx} {i} overflow!")
                r.rs[thread_idx][i] = (r.rs[thread_idx][i] * scale_factor) % mod
    return r

def test_ntt1():
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

    p1n2 = gpu_ntt_256(poly1, Q1, Q1_MONT, q1_twiddles)
    p2n2 = gpu_ntt_256(poly2,  Q1, Q1_MONT, q1_twiddles)
    p3n2 = [a * p2n2[idx] % Q1 for idx, a in enumerate(p1n2)]
    product2 = gpu_intt_256(p3n2,  Q1, Q1_MONT, q1_inv_twiddles)
    print(product2[:10])
    print(gpu_intt_256(p1n2, Q1, Q1_MONT, q1_inv_twiddles)[0:10])
    print(gpu_intt_256(p2n2, Q1, Q1_MONT, q1_inv_twiddles)[0:10])

def test_ntt_equivalent():
    raw = load_polys("../test_data/raw_poly.dat", False, 0, False)[0]
    crt0 = PolyHalf()
    crt1 = PolyHalf()
    for i in range(POLY_LEN):
        v = (raw.crt_0.data[i] << 32) + raw.crt_1.data[i]
        crt0.data[i] = v % Q1
        crt1.data[i] = v % Q2
    x = Poly(crt0, crt1)
    gpu_ntt_256(x.crt_0.data, Q1, Q1_MONT, q1_twiddles)
    gpu_ntt_256(x.crt_1.data, Q2, Q2_MONT, q2_twiddles)
    ntt_form = load_polys("../test_data/ntt_poly.dat", True, 0, False)[0]
    for i in range(POLY_LEN):
        if x.crt_0.data[i] % Q1 != ntt_form.crt_0.data[i] % Q1:
            print(f"crt0 {i} {x.crt_0.data[i]} {ntt_form.crt_0.data[i]}")
        if x.crt_1.data[i] % Q2 != ntt_form.crt_1.data[i] % Q2:
            print(f"crt1 {i} {x.crt_1.data[i]} {ntt_form.crt_1.data[i]}")
    gpu_intt_256(ntt_form.crt_0.data, Q1, Q1_MONT, q1_inv_twiddles, False)
    gpu_intt_256(ntt_form.crt_1.data, Q2, Q2_MONT, q2_inv_twiddles, False)
    inv_crt_poly(ntt_form)
    for i in range(POLY_LEN):
        v = (raw.crt_0.data[i] << 32) + raw.crt_1.data[i]
        v2 = (ntt_form.crt_0.data[i] << 32) + ntt_form.crt_1.data[i]
        if v % MODULUS != v2 % MODULUS: 
            print(f"raw {i} {v} {v2}")

def debug_ntt():
    raw = load_polys("../test_data/raw_poly.dat", False, 0, False)[0]
    crt0 = PolyHalf()
    crt1 = PolyHalf()
    for i in range(POLY_LEN):
        v = (raw.crt_0.data[i] << 32) + raw.crt_1.data[i]
        crt0.data[i] = v % Q1
        crt1.data[i] = v % Q2
    x = Poly(crt0, crt1)
    o = deepcopy(x)
    gpu_ntt_256(x.crt_0.data, Q1, Q1_MONT, q1_twiddles)
    gpu_intt_256(x.crt_0.data, Q1, Q1_MONT, q1_inv_twiddles, False)
    for i in range(POLY_LEN):
        x.crt_0.data[i] = (x.crt_0.data[i] * 268238881) % Q1
        if x.crt_0.data[i] != o.crt_0.data[i]:
            print(f"{i} {x.crt_0.data[i]} {o.crt_0.data[i]}")

def test_twiddle_ratio():
    indexes = [[] for _ in range(256)]
    for i in range(3,LEN_LOG2-3):
        for tid in range(256):
            # I wonder if you could compute this with one multiply from the previous value
            # Multiply by a if the bit now used is 0, and by b if the bit now used is 1
            twiddle_idx = (1 << i) + (tid >> (8 - i))
            indexes[tid].append(bitReverse(twiddle_idx, 11))
    for idx_set in indexes:
        ratios = []
        for i in range(1, len(idx_set)):
            # ratio = (q1_twiddles[idx_set[i]] * pow(q1_twiddles[idx_set[i - 1]], -1, Q1)) % Q1
            # ratio = (q1_inv_twiddles[idx_set[i]] * pow(q1_inv_twiddles[idx_set[i - 1]], -1, Q1)) % Q1
            ratio = idx_set[i] - idx_set[i - 1]
            ratios.append(ratio)
        print(ratios)

if __name__ == "__main__":
    test_ntt1()
    test_ntt_equivalent()
    debug_ntt()
    # test_twiddle_ratio()