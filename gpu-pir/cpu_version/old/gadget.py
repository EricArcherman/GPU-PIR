import copy
from params import *
from structures import *
from ntt import *
from arith import *

def from_ntt_inplace(x: Poly):
    # print(f"intt-in {test_hash(x.crt_0.data)}")
    intt_in_place(x.crt_0.data, Q1, Q1_MONT, q1_inv_twiddles, False)
    intt_in_place(x.crt_1.data, Q2, Q2_MONT, q2_inv_twiddles, False)
    # print(f"intt-out {test_hash(x.crt_0.data)}")
    inv_crt_poly(x)

# Changed to ceiling function
def get_bits_per(dim):
    return (MOD_BITS + dim - 1) // dim

# multiplication matrix
# expansion: 2 by t_exp_left or t_exp_right
# conversion: 2 by 2t_conv
# folding: 2 by 2t_gsw
# times
# input matrix
# expansion: 1 by 1 -> t_exp_left (or right) by 1
# conversion: 2 by 1 -> 2t_conv by 1
# folding: 2 by 1 -> 2t_gsw by 1
# equals and add to
# output accumulator
# expansion: 2 by 1
# conversion: 2 by 1
# folding: 2 by 1
def inv_gadget_ntt_mult_rdim_q1(input: Poly, acc: Ciphertext, mat: List[Ciphertext], j, rdim, mx, correct=False):
    # These constants should end up being inlined?  Could also just have a constant lookup table if bits_per division is expensive
    # Ideally these loops would be unrolled - but then there will be three copies of this function
    # Six copies if you split by CRT modulus as well
    num_elems = mx // rdim
    bits_per = get_bits_per(num_elems)
    mask = (1 << bits_per) - 1
    sm = Poly()
    for k in range(num_elems):
        row = j + k * rdim
        shift = k * bits_per
        for thread_idx in range(NUM_THREADS):
            for idx in range(0, POLY_LEN, NUM_THREADS):
                # Can reorganize data to avoid loading both and shifting/masking one element away
                # Or could just write output of from_ntt in bit-chunks that will be needed in the future
                z = idx + thread_idx
                high = input.crt_0.data[z]
                low = input.crt_1.data[z]
                combined = (high << 32) + low
                combined = combined >> shift
                combined = combined & mask
                sm.crt_0.data[z] = ensure32bit(combined)
                sm.crt_1.data[z] = ensure32bit(combined)
        # print(f"0 {j} {k} {test_hash(sm.crt_0.data)} {row} {shift}")
        ntt_in_place(sm.crt_0.data, Q1, Q1_MONT, q1_twiddles, correct)
        # i = 0
        mod_mult_add_poly(sm.crt_0.data, mat[row].a.crt_0.data, acc.a.crt_0.data, Q1, Q1_MONT, correct)
        # print(f"0 0 {row} {j} {k} {test_hash(mat[row].a.crt_0.data)} {test_hash(sm.crt_0.data)} {test_hash(acc.a.crt_0.data)}")
        # i = 1
        mod_mult_add_poly(sm.crt_0.data, mat[row].as_e.crt_0.data, acc.as_e.crt_0.data, Q1, Q1_MONT, correct)
        # print(f"1 0 {row} {j} {k} {test_hash(mat[row].as_e.crt_0.data)} {test_hash(sm.crt_0.data)} {test_hash(acc.as_e.crt_0.data)}")
def inv_gadget_ntt_mult_rdim_q2(input: Poly, acc: Ciphertext, mat: List[Ciphertext], j, rdim, mx, correct=False):
    num_elems = mx // rdim
    bits_per = get_bits_per(num_elems)
    mask = (1 << bits_per) - 1
    sm = Poly()
    for k in range(num_elems):
        row = j + k * rdim
        shift = k * bits_per
        for thread_idx in range(NUM_THREADS):
            for idx in range(0, POLY_LEN, NUM_THREADS):
                z = idx + thread_idx
                high = input.crt_0.data[z]
                low = input.crt_1.data[z]
                combined = (high << 32) + low
                combined = combined >> shift
                combined = combined & mask
                sm.crt_0.data[z] = ensure32bit(combined)
                sm.crt_1.data[z] = ensure32bit(combined)     
        ntt_in_place(sm.crt_1.data, Q2, Q2_MONT, q2_twiddles, correct)
        # i = 0
        mod_mult_add_poly(sm.crt_1.data, mat[row].a.crt_1.data, acc.a.crt_1.data, Q2, Q2_MONT, correct)
        # i = 1
        mod_mult_add_poly(sm.crt_1.data, mat[row].as_e.crt_1.data, acc.as_e.crt_1.data, Q2, Q2_MONT, correct)
        
def inv_gadget_ntt_mult(input: Ciphertext, acc: Ciphertext, mul: List[Ciphertext], mx):
    inv_gadget_ntt_mult_rdim_q1(input.a, acc, mul, 0, 2, mx, False)
    inv_gadget_ntt_mult_rdim_q2(input.a, acc, mul, 0, 2, mx, False)
    inv_gadget_ntt_mult_rdim_q1(input.as_e, acc, mul, 1, 2, mx, False)
    inv_gadget_ntt_mult_rdim_q2(input.as_e, acc, mul, 1, 2, mx, False)
    correct_one(acc.a)
    correct_one(acc.as_e)

def inv_gadget_ntt_mult_rdim1(input: Poly, acc: Ciphertext, mul: List[Ciphertext], mx):
    inv_gadget_ntt_mult_rdim_q1(input, acc, mul, 0, 1, mx, False)
    inv_gadget_ntt_mult_rdim_q2(input, acc, mul, 0, 1, mx, False)
    correct_one(acc.a)
    correct_one(acc.as_e)
