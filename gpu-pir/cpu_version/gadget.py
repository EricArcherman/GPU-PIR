from params import *
from structures import *
from ntt import *
from arith import *
from utils import *

def from_ntt_reg(q1_values: MultiRegisters, q2_values: MultiRegisters):
    # Async load Q1 values?
    # Can do INTT on Q1 after computing it
    # Where to store/retrieve to the values though?
    # I have room in registers I think if I can skip loading through shared memory
    # Idk if it waits for loads to finish or only when it next needs the value
    for thread_idx in range(NUM_THREADS):
        for i in range(2*NUM_PAIRS):
            q1_values.rs[thread_idx][i] -= Q1 * (q1_values.rs[thread_idx][i] >= Q1)
            q2_values.rs[thread_idx][i] -= Q2 * (q2_values.rs[thread_idx][i] >= Q2)
    # print(f"intt-in {test_hash(combine(q1_values))}")
    intt_regs(q1_values, Q1, Q1_MONT, q1_inv_twiddles, False)
    # test_hash_reduce(q1_values.rs, "q1 intt")
    intt_regs(q2_values, Q2, Q2_MONT, q2_inv_twiddles, False)
    # test_hash_reduce(q2_values.rs, "q2 intt")
    # print(f"intt-out {test_hash(raw_combine(q1_values))}")
    inv_crt_regs(q1_values, q2_values)
    # test_hash_reduce(q1_values.rs, "q1 crt")
    # test_hash_reduce(q2_values.rs, "q2 crt")

def write_regs(dest: Poly, q1_values: MultiRegisters, q2_values: MultiRegisters):
    for thread_idx in range(NUM_THREADS):
        for (i, idx) in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            z = thread_idx + idx
            dest.crt_0.data[z] = q1_values.rs[thread_idx][i]
            dest.crt_1.data[z] = q2_values.rs[thread_idx][i]

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
def inv_gadget_ntt_mult1(twiddles, input: Poly, acc_a: MultiRegisters, acc_as_e: MultiRegisters, mat: List[CiphertextHalf], j, rdim, num_elems, bits_per, mask, q, q_inv, correct=False):
    # These constants should end up being inlined?  Could also just have a constant lookup table if bits_per division is expensive
    # Ideally these loops would be unrolled - but then there will be three copies of this function
    # Six copies if you split by CRT modulus as well
    for k in range(num_elems):
        row = j + k * rdim
        shift = k * bits_per
        r = MultiRegisters()
        for thread_idx in range(NUM_THREADS):
            for (i, idx) in enumerate(range(0, POLY_LEN, NUM_THREADS)):
                # Can reorganize data to avoid loading both and shifting/masking one element away
                # Or could just write output of from_ntt in bit-chunks that will be needed in the future
                z = idx + thread_idx
                high = input.crt_0.data[z]
                low = input.crt_1.data[z]
                combined = (high << 32) + low
                combined = combined >> shift
                combined = combined & mask
                r.rs[thread_idx][i] = ensure32bit(combined)
        # print(f"0 {j} {k} {test_hash(raw_combine(r))} {row} {shift}")
        # test_hash_reduce(r.rs, "input")
        ntt_regs(r, q, q_inv, twiddles)
        # test_hash_reduce(r.rs, "ntt")
        mod_mult_add_regs(r, mat[row].a, acc_a, q, q_inv, correct)
        # test_hash_reduce(acc_a.rs, "add")
        # print(f"0 0 {row} {j} {k} {test_hash(mat[row].a.data)} {test_hash(combine(r))} {test_hash(combine(acc_a))}")
        mod_mult_add_regs(r, mat[row].as_e, acc_as_e, q, q_inv, correct)
        # print(f"1 0 {row} {j} {k} {test_hash(mat[row].as_e.crt_0.data)} {test_hash(sm.crt_0.data)} {test_hash(acc.as_e.crt_0.data)}")

def inv_gadget_ntt_mult_rdim2(twiddles, input: Ciphertext, acc_a: MultiRegisters, acc_as_e: MultiRegisters, mul: List[CiphertextHalf], mx, q, q_inv, correct=False):
    rdim = 2
    num_elems = mx // rdim
    bits_per = get_bits_per(num_elems)
    mask = (1 << bits_per) - 1
    inv_gadget_ntt_mult1(twiddles, input.a, acc_a, acc_as_e, mul, 0, rdim, num_elems, bits_per, mask, q, q_inv, correct)
    inv_gadget_ntt_mult1(twiddles, input.as_e, acc_a, acc_as_e, mul, 1, rdim, num_elems, bits_per, mask, q, q_inv, correct)

def inv_gadget_ntt_mult_rdim1(twiddles, input: Ciphertext, acc_a: MultiRegisters, acc_as_e: MultiRegisters, mul: List[CiphertextHalf], mx, q, q_inv, correct=False):
    rdim = 1
    num_elems = mx // rdim
    bits_per = get_bits_per(num_elems)
    mask = (1 << bits_per) - 1
    inv_gadget_ntt_mult1(twiddles, input.a, acc_a, acc_as_e, mul, 0, rdim, num_elems, bits_per, mask, q, q_inv, correct)