import random
from sympy import ntt, intt
from params import *
from arith import *
from testing import *
import numpy as np

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

q1_twiddles = to_mont(generate_twiddles(POLY_LEN, Q1_ROOT, Q1), Q1)
q2_twiddles = to_mont(generate_twiddles(POLY_LEN, Q2_ROOT, Q2), Q2)
q1_inv_twiddles = to_mont(generate_twiddles(POLY_LEN, Q1_ROOT_INV, Q1), Q1)
q2_inv_twiddles = to_mont(generate_twiddles(POLY_LEN, Q2_ROOT_INV, Q2), Q2)
q1_corr = pow(POLY_LEN, -1, Q1)
q2_corr = pow(POLY_LEN, -1, Q2)

def ntt_in_place(shared, modulus, modulus_inv, twiddles, correct=True):
    gpu_ntt(shared, modulus, modulus_inv, twiddles, correct)

def intt_in_place(shared, modulus, modulus_inv, inv_twiddles, correct=True):
    gpu_intt(shared, modulus, modulus_inv, inv_twiddles, correct)

# Radix-16
# Remove syncs
# Remove corrections
# Remove twiddle when it is 1

# Set twiddles from negations and bake in negative

# Can just bake the inverse correction into the crt constant
# Can also bake the montgomery out value as well 
def gpu_ntt(shared, modulus, inv_modulus, twiddles, correct=True):
    for i in range(11):
        length = 1 << i
        step = (2048//length)//2
        for local_tid in range(1024):
            psi_step = local_tid // step
            target_index = psi_step * step * 2 + local_tid % step
            psi = twiddles[length + psi_step]
            # print(f"{i} {local_tid} {target_index} {target_index + step} {psi_step} {length + psi_step} {((target_index) >> (11-i))}")
            x = shared[target_index]
            y = shared[target_index + step]
            t = montgomery_mult(y, psi, modulus, inv_modulus, False)
            if i == 8:
                if x >= 8*modulus:
                    x -= 8*modulus
            x_prime = x + t
            y_prime = x - t + 2*modulus
            shared[target_index] = x_prime
            shared[target_index + step] = y_prime
            # print(f"{i} {target_index} {target_index + step} {length + psi_step} {psi} {x % modulus} {y % modulus} {x_prime % modulus} {y_prime % modulus}")
        sm = [x % modulus for x in shared]
        # print(f"{i} {test_hash(sm)}")
    if correct:
        for i in range(len(shared)):
            if shared[i] >= 2**32:
                print("overflow!")
            shared[i] = shared[i] % modulus
    return shared

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

def swap_fix(r):
    # r[0] = r[0]
    # r[2] = r[1]
    # r[4] = r[2]
    # r[6] = r[3]
    # r[8] = r[4]
    # r[10] = r[5]
    # r[12] = r[6]
    # r[14] = r[7]
    # r[1] = r[8]
    # r[3] = r[9]
    # r[5] = r[10]
    # r[7] = r[11]
    # r[9] = r[12]
    # r[11] = r[13]
    # r[13] = r[14]
    # r[15] = r[15]
    tmp_2 = r[2]
    r[2] = r[1]
    tmp_4 = r[4]
    r[4] = tmp_2
    tmp_6 = r[6]
    r[6] = r[3]
    tmp_8 = r[8]
    r[8] = tmp_4
    tmp_10 = r[10]
    r[10] = r[5]
    tmp_12 = r[12]
    r[12] = tmp_6
    tmp_14 = r[14]
    r[14] = r[7]
    r[1] = tmp_8
    tmp_9 = r[9]
    r[3] = tmp_9
    r[5] = tmp_10
    r[7] = r[11]
    r[9] = tmp_12
    r[11] = r[13]
    r[13] = tmp_14

def swap_fix_n(r):
    r_tmp = Registers()
    for i in range(RADIX//2):
        r_tmp[2*i] = r[i]
    for i in range(RADIX//2):
        r_tmp[2*i + 1] = r[i + RADIX//2]
    r = r_tmp

def butterfly(r, x_idx, y_idx, psi, modulus, inv_modulus):
    x = r[x_idx]
    y = r[y_idx]
    t = montgomery_mult(y, psi, modulus, inv_modulus, False)
    r[x_idx] = x + t
    r[y_idx] = x - t + 2*modulus

# x and y are matched if x and y differ by 1 bit in the i-th place
# the twiddle is indexed by the common prefix before the i-th bit and i, and is g^(bit reversed prefix) (where y includes a 1 bit in the i-th position)
# For every pair (x,y) in the i'th level there is a partner pair (x', y') such that
# (x', y') is a pair, and shares the same i-th prefix, and i+2-th suffix,
# in the (i+1)th level, (x, x') and (y, y') are pairs
# Therefore it suffices to swap y for x' using a warp-shuffle.'

# Sequential ordering - thread gets first consecutive x's
# Strided ordering - threads iterate over x's
# Something in between - because prefixes jump at i-s

# Twiddle reuse - within a thread (reduce loadings)
# Twiddle reuse - across a warp (constant memory loading)

# f_i(thread_id) -> x's needed.  Calc y's needed
# for each x,y Calc f_(i-1)^-1(x,y) -> thread that has x,y value.  If my thread, get from my registers, else, get from other thread (presumably through a swap)

def twiddle_idx(i, x_idx):
    return (1 << i) + (x_idx >> (11 - i))

def butterfly(x_val, y_val, x_idx, i, twiddles, modulus, inv_modulus):
    psi = twiddles[(1 << i) + (x_idx >> (11 - i))]
    t = montgomery_mult(y_val, psi, modulus, inv_modulus, False)
    return x_val + t, x_val - t + 2*modulus

def align(i, idx):
    p = 1 << (10 - i)
    x = ((idx >> (10 - i)) << (11 - i)) ^ (idx & (((1 << (10 - i)) - 1)))
    y = x ^ p
    return x, y

def seq_str(i, seq_len, stride_log2, thread_id):
    stride_size = 1 << stride_log2
    batch = thread_id >> stride_log2
    entry = thread_id & ((1 << stride_log2) - 1)
    x = batch * stride_size * seq_len + entry
    out = []
    for _ in range(seq_len):
        out.append(align(i, x))
        x += stride_size
    return out

def assign(i, thread_id):
    seq_len = (POLY_LEN // NUM_THREADS) // 2
    radix_log2 = 3
    if i < radix_log2:
        return seq_str(i, seq_len, 11-radix_log2, thread_id)
    elif i < 2*radix_log2:
        return seq_str(i, seq_len, 11-(2*radix_log2), thread_id)
    else:
        return seq_str(i, seq_len, 11-i, thread_id)

def xor_ntt(r: Registers, modulus, inv_modulus, twiddle_root):
    before = 5
    after = 11 - before
    # Calculate twiddle values here
    # Precompute in GPU code
    # By storing the values from a CPU computation



# NTT for 2048
# Input 16 values spaces by 128
# Output 16 values consecutive
def f(r,modulus,inv_modulus,twiddles):
    # Stride 1024
    for tid in range(128):
        butterfly(r[tid], 0, 8, twiddles[1], modulus, inv_modulus)
        butterfly(r[tid], 1, 9, twiddles[1], modulus, inv_modulus)
        butterfly(r[tid], 2, 10, twiddles[1], modulus, inv_modulus)
        butterfly(r[tid], 3, 11, twiddles[1], modulus, inv_modulus)
        butterfly(r[tid], 4, 12, twiddles[1], modulus, inv_modulus)
        butterfly(r[tid], 5, 13, twiddles[1], modulus, inv_modulus)
        butterfly(r[tid], 6, 14, twiddles[1], modulus, inv_modulus)
        butterfly(r[tid], 7, 15, twiddles[1], modulus, inv_modulus)
    # Stride 512
    for tid in range(128):
        butterfly(r[tid], 0, 4, twiddles[2], modulus, inv_modulus)
        butterfly(r[tid], 1, 5, twiddles[2], modulus, inv_modulus)
        butterfly(r[tid], 2, 6, twiddles[2], modulus, inv_modulus)
        butterfly(r[tid], 3, 7, twiddles[2], modulus, inv_modulus)
        butterfly(r[tid], 8, 12, twiddles[3], modulus, inv_modulus)
        butterfly(r[tid], 9, 13, twiddles[3], modulus, inv_modulus)
        butterfly(r[tid], 10, 14, twiddles[3], modulus, inv_modulus)
        butterfly(r[tid], 11, 15, twiddles[3], modulus, inv_modulus)
    # Stride 256
    for tid in range(128):
        butterfly(r[tid], 0, 2, twiddles[4], modulus, inv_modulus)
        butterfly(r[tid], 1, 3, twiddles[4], modulus, inv_modulus)
        butterfly(r[tid], 4, 6, twiddles[5], modulus, inv_modulus)
        butterfly(r[tid], 5, 7, twiddles[5], modulus, inv_modulus)
        butterfly(r[tid], 8, 10, twiddles[6], modulus, inv_modulus)
        butterfly(r[tid], 9, 11, twiddles[6], modulus, inv_modulus)
        butterfly(r[tid], 12, 14, twiddles[7], modulus, inv_modulus)
        butterfly(r[tid], 13, 15, twiddles[7], modulus, inv_modulus)
    # Stride 128
    for tid in range(128):
        butterfly(r[tid], 0, 1, twiddles[8], modulus, inv_modulus)
        butterfly(r[tid], 2, 3, twiddles[9], modulus, inv_modulus)
        butterfly(r[tid], 4, 5, twiddles[10], modulus, inv_modulus)
        butterfly(r[tid], 6, 7, twiddles[11], modulus, inv_modulus)
        butterfly(r[tid], 8, 9, twiddles[12], modulus, inv_modulus)
        butterfly(r[tid], 10, 11, twiddles[13], modulus, inv_modulus)
        butterfly(r[tid], 12, 13, twiddles[14], modulus, inv_modulus)
        butterfly(r[tid], 14, 15, twiddles[15], modulus, inv_modulus)
    # Here we use shared memory to swap values between warps
    shared = [0 for _ in range(2048)]
    for tid in range(128):
        for i in range(16):
            shared[i*128 + tid] = r[tid][i]
    # __syncthreads()
    # yield
    # Read from shared memory: TODO: fix bank conflicts
    # Read 16 values at stride 8
    for tid in range(128):
        u = tid >> 3
        l = tid & 7
        for i in range(16):
            r[tid][i] = shared[u*128+i*8+l]
    # Stride 64
    for tid in range(128):
        twiddle_idx = 16 + (tid >> 3)
        butterfly(r[tid], 0, 8, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 1, 9, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 2, 10, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 3, 11, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 4, 12, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 5, 13, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 6, 14, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 7, 15, twiddles[twiddle_idx], modulus, inv_modulus)
    # After swap: 16 values at stride 4
    # No parallelism in python
    swapped = [[0 for _ in range(16)] for _ in range(2048)]
    for tid in range(128):
        butterfly_swap(r, swapped, 0, 8, tid, 4)
        butterfly_swap(r, swapped, 1, 9, tid, 4)
        butterfly_swap(r, swapped, 2, 10, tid, 4)
        butterfly_swap(r, swapped, 3, 11, tid, 4)
        butterfly_swap(r, swapped, 4, 12, tid, 4)
        butterfly_swap(r, swapped, 5, 13, tid, 4)
        butterfly_swap(r, swapped, 6, 14, tid, 4)
        butterfly_swap(r, swapped, 7, 15, tid, 4)
    r = swapped
    # My brain hurts
    for tid in range(128):
        swap_fix(r[tid])
    
    # # Stride 32
    for tid in range(128):
        twiddle_idx = 32 + (tid >> 2)
        # u = tid >> 2
        # l = tid & 3
        # for i in range(8):
        #     t1 = i
        #     t2 = i + 4
        #     g1 = u*64+t1*4+l
        #     g2 = u*64+t2*4+l
        #     print(f"{tid} {g1} {g2} {twiddle_idx - 32}")
        butterfly(r[tid], 0, 8, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 1, 9, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 2, 10, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 3, 11, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 4, 12, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 5, 13, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 6, 14, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 7, 15, twiddles[twiddle_idx], modulus, inv_modulus)
    
    # # After swap: 16 values at stride 2
    # # No parallelism in python
    swapped = [[0 for _ in range(16)] for _ in range(2048)]
    for tid in range(128):
        butterfly_swap(r, swapped, 0, 8, tid, 2)
        butterfly_swap(r, swapped, 1, 9, tid, 2)
        butterfly_swap(r, swapped, 2, 10, tid, 2)
        butterfly_swap(r, swapped, 3, 11, tid, 2)
        butterfly_swap(r, swapped, 4, 12, tid, 2)
        butterfly_swap(r, swapped, 5, 13, tid, 2)
        butterfly_swap(r, swapped, 6, 14, tid, 2)
        butterfly_swap(r, swapped, 7, 15, tid, 2)
    r = swapped
    # My brain hurts
    for tid in range(128):
        swap_fix(r[tid])

    # Can apply this modulus correction anywhere between (5 and 8) of the 11 iterations
    for tid in range(128):
        for i in range(8):
            r[tid][i] -= modulus * (r[tid][i] >= (8 * modulus))

    # Stride 16
    for tid in range(128):
        twiddle_idx = 64 + (tid >> 1)
        # u = tid >> 1
        # l = tid & 1
        # for i in range(16):
        #     shared[u*32+i*2+l] = r[tid][i]
        butterfly(r[tid], 0, 8, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 1, 9, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 2, 10, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 3, 11, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 4, 12, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 5, 13, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 6, 14, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 7, 15, twiddles[twiddle_idx], modulus, inv_modulus)

    # After swap: 16 values at stride 1
    # No parallelism in python
    swapped = [[0 for _ in range(16)] for _ in range(2048)]
    for tid in range(128):
        butterfly_swap(r, swapped, 0, 8, tid, 1)
        butterfly_swap(r, swapped, 1, 9, tid, 1)
        butterfly_swap(r, swapped, 2, 10, tid, 1)
        butterfly_swap(r, swapped, 3, 11, tid, 1)
        butterfly_swap(r, swapped, 4, 12, tid, 1)
        butterfly_swap(r, swapped, 5, 13, tid, 1)
        butterfly_swap(r, swapped, 6, 14, tid, 1)
        butterfly_swap(r, swapped, 7, 15, tid, 1)
    r = swapped
    # My brain hurts
    for tid in range(128):
        swap_fix(r[tid])

    # Stride 8
    for tid in range(128):
        twiddle_idx = 128 + tid
        butterfly(r[tid], 0, 8, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 1, 9, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 2, 10, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 3, 11, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 4, 12, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 5, 13, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 6, 14, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 7, 15, twiddles[twiddle_idx], modulus, inv_modulus)
    # Stride 4
    for tid in range(128):
        twiddle_idx = 256 + 2 * tid
        butterfly(r[tid], 0, 4, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 1, 5, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 2, 6, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 3, 7, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 8, 12, twiddles[twiddle_idx+1], modulus, inv_modulus)
        butterfly(r[tid], 9, 13, twiddles[twiddle_idx+1], modulus, inv_modulus)
        butterfly(r[tid], 10, 14, twiddles[twiddle_idx+1], modulus, inv_modulus)
        butterfly(r[tid], 11, 15, twiddles[twiddle_idx+1], modulus, inv_modulus)
    # Stride 2
    for tid in range(128):
        twiddle_idx = 512 + 4 * tid
        butterfly(r[tid], 0, 2, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 1, 3, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 4, 6, twiddles[twiddle_idx+1], modulus, inv_modulus)
        butterfly(r[tid], 5, 7, twiddles[twiddle_idx+1], modulus, inv_modulus)
        butterfly(r[tid], 8, 10, twiddles[twiddle_idx+2], modulus, inv_modulus)
        butterfly(r[tid], 9, 11, twiddles[twiddle_idx+2], modulus, inv_modulus)
        butterfly(r[tid], 12, 14, twiddles[twiddle_idx+3], modulus, inv_modulus)
        butterfly(r[tid], 13, 15, twiddles[twiddle_idx+3], modulus, inv_modulus)
    # Stride 1
    for tid in range(128):
        twiddle_idx = 1024 + 8 * tid
        butterfly(r[tid], 0, 1, twiddles[twiddle_idx], modulus, inv_modulus)
        butterfly(r[tid], 2, 3, twiddles[twiddle_idx+1], modulus, inv_modulus)
        butterfly(r[tid], 4, 5, twiddles[twiddle_idx+2], modulus, inv_modulus)
        butterfly(r[tid], 6, 7, twiddles[twiddle_idx+3], modulus, inv_modulus)
        butterfly(r[tid], 8, 9, twiddles[twiddle_idx+4], modulus, inv_modulus)
        butterfly(r[tid], 10, 11, twiddles[twiddle_idx+5], modulus, inv_modulus)
        butterfly(r[tid], 12, 13, twiddles[twiddle_idx+6], modulus, inv_modulus)
        butterfly(r[tid], 14, 15, twiddles[twiddle_idx+7], modulus, inv_modulus)
    return r

def gpu_ntt_radix16(shared, modulus, inv_modulus, twiddles, correct=True):
    r = [[0 for _ in range(16)] for _ in range(128)]
    # copy global to local - likely input will already be in registers from bitshift operation
    for thread_idx in range(0, 128):
        for l, i in enumerate(range(0, 2048, 128)):
            r[thread_idx][l] = shared[i + thread_idx]
    r = f(r, modulus, inv_modulus, twiddles)
    for thread_idx in range(0, 128):
        for i in range(16):
            shared[thread_idx*16 + i] = r[thread_idx][i]
    if correct:
        for i in range(len(shared)):
            if shared[i] >= 2**32:
                print(f"{i} overflow!")
            shared[i] = shared[i] % modulus
    return shared

# The correction factor will be combined into the inverse CRT
def gpu_intt(shared, modulus, inv_modulus, twiddles, correct=True):
    for i in range(10,-1,-1):
        length = 1 << i
        step = (2048//length)//2
        for local_tid in range(1024):
            psi_step = local_tid // step
            target_index = psi_step * step * 2 + local_tid % step
            psi = twiddles[length + psi_step]
            x = shared[target_index]
            y = shared[target_index + step]
            x_prime = x + y
            x_prime -= modulus * (x_prime >= modulus)
            shared[target_index] = x_prime
            t = x + modulus - y
            y_prime = montgomery_mult(psi, t, modulus, inv_modulus, False)
            y_prime -= modulus * (y_prime >= modulus)
            shared[target_index + step] = y_prime
            sm = [x % modulus for x in shared]
        # print(f"{i} {test_hash(sm)}")
    if correct:
        scale_factor = q1_corr if modulus == Q1 else q2_corr
        shared = [(a * scale_factor) % modulus  for a in shared]
    return shared

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

    p1n2 = gpu_ntt(poly1, Q1, Q1_MONT, q1_twiddles)
    p2n2 = gpu_ntt(poly2,  Q1, Q1_MONT, q1_twiddles)
    p3n2 = [a * p2n2[idx] % Q1 for idx, a in enumerate(p1n2)]
    product2 = gpu_intt(p3n2,  Q1, Q1_MONT, q1_inv_twiddles)
    print(product2[:10])
    print(gpu_intt(p1n2, Q1, Q1_MONT, q1_inv_twiddles)[0:10])
    print(gpu_intt(p2n2, Q1, Q1_MONT, q1_inv_twiddles)[0:10])

def test_ntt2():
    poly1 = [0 for _ in range(POLY_LEN)]
    poly2 = [0 for _ in range(POLY_LEN)]
    poly1[0] = Q2 - 1
    poly1[1] = 1
    for i in range(5):
        poly2[i] = 1
    p1n1 = ntt(poly1, Q2)
    p2n1 = ntt(poly2, Q2)
    p3n1 = [a * p2n1[idx] % Q2 for idx, a in enumerate(p1n1)]
    product1 = intt(p3n1, Q2)
    print(product1[:10])

    p1n2 = gpu_ntt(poly1, Q2, Q2_MONT, q2_twiddles)
    p2n2 = gpu_ntt(poly2,  Q2, Q2_MONT, q2_twiddles)
    p3n2 = [a * p2n2[idx] % Q2 for idx, a in enumerate(p1n2)]
    product2 = gpu_intt(p3n2,  Q2, Q2_MONT, q2_inv_twiddles)
    print(product2[:10])
    print(gpu_intt(p1n2, Q2, Q2_MONT, q2_inv_twiddles)[0:10])
    print(gpu_intt(p2n2, Q2, Q2_MONT, q2_inv_twiddles)[0:10])

def test_ntt_equivalent():
    raw = load_polys("raw_poly.dat", False, 0, False)[0]
    crt0 = PolyHalf()
    crt1 = PolyHalf()
    for i in range(POLY_LEN):
        v = (raw.crt_0.data[i] << 32) + raw.crt_1.data[i]
        crt0.data[i] = v % Q1
        crt1.data[i] = v % Q2
    x = Poly(crt0, crt1)
    gpu_ntt(x.crt_0.data, Q1, Q1_MONT, q1_twiddles)
    gpu_ntt(x.crt_1.data, Q2, Q2_MONT, q2_twiddles)
    ntt_form = load_polys("ntt_poly.dat", True, 0, False)[0]
    for i in range(POLY_LEN):
        if x.crt_0.data[i] % Q1 != ntt_form.crt_0.data[i] % Q1:
            print(f"crt0 {i} {x.crt_0.data[i]} {ntt_form.crt_0.data[i]}")
        if x.crt_1.data[i] % Q2 != ntt_form.crt_1.data[i] % Q2:
            print(f"crt1 {i} {x.crt_1.data[i]} {ntt_form.crt_1.data[i]}")
    gpu_intt(ntt_form.crt_0.data, Q1, Q1_MONT, q1_inv_twiddles, False)
    gpu_intt(ntt_form.crt_1.data, Q2, Q2_MONT, q2_inv_twiddles, False)
    inv_crt_poly(ntt_form)
    for i in range(POLY_LEN):
        v = (raw.crt_0.data[i] << 32) + raw.crt_1.data[i]
        v2 = (ntt_form.crt_0.data[i] << 32) + ntt_form.crt_1.data[i]
        if v % MODULUS != v2 % MODULUS: 
            print(f"raw {i} {v} {v2}")

def print_intt_pattern():
    # Arrange threads to access twiddles with different stride
    # Or arrange threads so a warp needs the fewest twiddles possible (and then access with different stride )
    for i in range(10,6,-1):
        length = 1 << i
        step = (2048//length)//2
        for local_tid in range(16):
            psi_step = local_tid // step
            target_index = psi_step * step * 2 + local_tid % step
            print(f"{target_index} {target_index + step} {length + psi_step}")

def print_ntt_pattern():
    for i in range(4):
        length = 1 << i
        step = (2048//length)//2
        print(f"{i} {step}")
        for local_tid in range(0,1024,128):
            psi_step = local_tid // step
            target_index = psi_step * step * 2 + local_tid % step
            print(f"{target_index} {target_index + step} {length + psi_step}")

    for i in range(4):
        length = 1 << i
        step = (2048//length)//2
        for local_tid in range(1,1024,128):
            psi_step = local_tid // step
            target_index = psi_step * step * 2 + local_tid % step
            print(f"{target_index} {target_index + step} {length + psi_step}")

    for i in range(4):
        length = 1 << i
        step = (2048//length)//2
        for local_tid in range(127,1024,128):
            psi_step = local_tid // step
            target_index = psi_step * step * 2 + local_tid % step
            print(f"{target_index} {target_index + step} {length + psi_step}")
    for i in range(11-4, 11):
        length = 1 << i
        step = (2048//length)//2
        print(f"{i} {step}")
        for local_tid in range(0,8):
            psi_step = local_tid // step
            target_index = psi_step * step * 2 + local_tid % step
            print(f"{target_index} {target_index + step} {length + psi_step}")
    for i in range(11-4, 11):
        length = 1 << i
        step = (2048//length)//2
        print(f"{i} {step}")
        for local_tid in range(1024-8,1024):
            psi_step = local_tid // step
            target_index = psi_step * step * 2 + local_tid % step
            print(f"{target_index} {target_index + step} {length + psi_step}")

def print_sm_pattern():
    assignments = [[-1 for _ in range(11)] for _ in range(2048)]
    for thread_idx in range(128):
        for i in range(4):
            length = 1 << i
            step = (2048//length)//2
            for j in range(8):
                local_tid = thread_idx + j * 128
                psi_step = local_tid // step
                target_index = psi_step * step * 2 + local_tid % step
                assignments[target_index][i] = thread_idx
                assignments[target_index+step][i] = thread_idx
        for i in range(4,7):
            length = 1 << i
            step = (2048//length)//2
            for j in range(4):
                local_tid = (thread_idx >> 4) * 64 + thread_idx % 16 + j*16
                psi_step = local_tid // step
                target_index = psi_step * step * 2 + local_tid % step
                assignments[target_index][i] = thread_idx
                assignments[target_index+step][i] = thread_idx
            for j in range(4):
                local_tid = 512 + (thread_idx >> 4) * 64 + thread_idx % 16 + j*16
                psi_step = local_tid // step
                target_index = psi_step * step * 2 + local_tid % step
                assignments[target_index][i] = thread_idx
                assignments[target_index+step][i] = thread_idx
        for i in range(7,11):
            length = 1 << i
            step = (2048//length)//2
            for j in range(8):
                local_tid = thread_idx * 8 + j
                psi_step = local_tid // step
                target_index = psi_step * step * 2 + local_tid % step
                assignments[target_index][i] = thread_idx
                assignments[target_index+step][i] = thread_idx
    for a in assignments:
        print(a)

def print_bit_reversal():
    for i in range(2048):
        print(f"{i} {bitReverse(i, 11)}")

def twiddle_reuse_sum():
    for i in range(POLY_LEN):
        for (idx, t) in enumerate(q1_inv_twiddles):
            if (t == Q1 - q1_twiddles[i]):
                # The sum is 3*2^n -1.  n is the number of bits - hopefully one of i j or k
                print(f"{i} -> {idx} , {i + idx}")

def test_radix16():
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

    p1n2 = gpu_ntt_radix16(poly1, Q1, Q1_MONT, q1_twiddles)
    p2n2 = gpu_ntt_radix16(poly2,  Q1, Q1_MONT, q1_twiddles)
    p3n2 = [a * p2n2[idx] % Q1 for idx, a in enumerate(p1n2)]
    product2 = gpu_intt(p3n2,  Q1, Q1_MONT, q1_inv_twiddles)
    print(product2[:10])
    print(gpu_intt(p1n2, Q1, Q1_MONT, q1_inv_twiddles)[0:10])
    print(gpu_intt(p2n2, Q1, Q1_MONT, q1_inv_twiddles)[0:10])
    r = [random.randint(0, Q1) for _ in range(POLY_LEN)]
    r_copy = [x for x in r]
    r = gpu_ntt_radix16(r, Q1, Q1_MONT, q1_twiddles)
    r = gpu_intt(r, Q1, Q1_MONT, q1_inv_twiddles)
    for i in range(POLY_LEN):
        if r[i] != r_copy[i]:
            print(f"Fail {i} {r[i]} {r_copy[i]}")
            break

def ntt16(shared, modulus, inv_modulus, twiddles):
    for i in range(7,11):
        length = 1 << i
        step = (2048//length)//2
        for local_tid in range(8):
            psi_step = local_tid // step
            target_index = psi_step * step * 2 + local_tid % step
            psi = twiddles[length + psi_step]
            print(f"{i-7} {local_tid}: {target_index} <-> {target_index + step} {psi_step} w^{length // 128 + psi_step}")
            x = shared[target_index]
            y = shared[target_index + step]
            t = montgomery_mult(y, psi, modulus, inv_modulus, False)
            if i == 7:
                if x >= 8*modulus:
                    x -= 8*modulus
            x_prime = x + t
            y_prime = x - t + 2*modulus
            shared[target_index] = x_prime
            shared[target_index + step] = y_prime
    return shared

def test_butterfly_swap():
    shared = [i for i in range(2048)]
    r = [[0 for _ in range(16)] for _ in range(128)]
    for tid in range(128):
        u = tid >> 3
        l = tid & 7
        for i in range(16):
            r[tid][i] = shared[u*128+i*8+l]
    
    print(f"Input (by 8): {r[:10]}")

    swapped = [[0 for _ in range(16)] for _ in range(128)]
    for tid in range(128):
        butterfly_swap(r, swapped, 0, 8, tid, 4)
        butterfly_swap(r, swapped, 1, 9, tid, 4)
        butterfly_swap(r, swapped, 2, 10, tid, 4)
        butterfly_swap(r, swapped, 3, 11, tid, 4)
        butterfly_swap(r, swapped, 4, 12, tid, 4)
        butterfly_swap(r, swapped, 5, 13, tid, 4)
        butterfly_swap(r, swapped, 6, 14, tid, 4)
        butterfly_swap(r, swapped, 7, 15, tid, 4)
    r = swapped

    print(f"(Transferred): {r[:10]}")

    # My brain hurts
    for tid in range(128):
        swap_fix(r[tid])

    print(f"(By 4): {r[:10]}")

    swapped = [[0 for _ in range(16)] for _ in range(128)]
    for tid in range(128):
        butterfly_swap(r, swapped, 0, 8, tid, 2)
        butterfly_swap(r, swapped, 1, 9, tid, 2)
        butterfly_swap(r, swapped, 2, 10, tid, 2)
        butterfly_swap(r, swapped, 3, 11, tid, 2)
        butterfly_swap(r, swapped, 4, 12, tid, 2)
        butterfly_swap(r, swapped, 5, 13, tid, 2)
        butterfly_swap(r, swapped, 6, 14, tid, 2)
        butterfly_swap(r, swapped, 7, 15, tid, 2)
    r = swapped

    print(f"(Transferred): {r[:10]}")

    # My brain hurts
    for tid in range(128):
        swap_fix(r[tid])

    print(f"(By 2): {r[:10]}")

    swapped = [[0 for _ in range(16)] for _ in range(128)]
    for tid in range(128):
        butterfly_swap(r, swapped, 0, 8, tid, 1)
        butterfly_swap(r, swapped, 1, 9, tid, 1)
        butterfly_swap(r, swapped, 2, 10, tid, 1)
        butterfly_swap(r, swapped, 3, 11, tid, 1)
        butterfly_swap(r, swapped, 4, 12, tid, 1)
        butterfly_swap(r, swapped, 5, 13, tid, 1)
        butterfly_swap(r, swapped, 6, 14, tid, 1)
        butterfly_swap(r, swapped, 7, 15, tid, 1)
    r = swapped

    print(f"(Transferred): {r[:10]}")

    # My brain hurts
    for tid in range(128):
        swap_fix(r[tid])

    print(f"(By 1): {r[:10]}")

if __name__ == "__main__":
    # test_ntt1()
    # test_ntt2()
    # print_sm_pattern()
    # print_ntt_pattern()
    # test_radix16()
    # test_ntt1()
    # test_butterfly_swap()
    test_ntt1()
    test_ntt_equivalent()