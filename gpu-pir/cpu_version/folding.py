from params import *
from structures import *
from gadget import *
from arith import *
from testing import *

def v_folding_neg(v_folding: List[Poly]):
    # I don't understand why you have to be in raw form to negate
    # It should be NTT(0 - x) = NTT(0) - NTT(x) = 0 - NTT(x)?

    # Have gadget stored in memory and transferred from CPU during twiddle setup
    return v_folding


# Ideally I convert each regev to gsw one row at a time
# Use that row immediately for all foldings with this query
# Negate that row immediately
# Use that row immediately for the negated side
def fold_ciphertexts(accumulators: List[Ciphertext], v_folding: List[List[List[CiphertextHalf]]], v_folding_neg: List[List[List[CiphertextHalf]]], num_per, i, cur_dim):
   
    # Zero initialized
    acc_a_q1 = MultiRegisters()
    acc_as_e_q1 = MultiRegisters()
    inv_gadget_ntt_mult_rdim2(q1_twiddles, accumulators[num_per + i], acc_a_q1, acc_as_e_q1, v_folding[V2 - 1 - cur_dim][0], 2*T_GSW, Q1, Q1_MONT)
    print(f"Product {test_hash(combine(acc_a_q1))}")
    inv_gadget_ntt_mult_rdim2(q1_twiddles, accumulators[i], acc_a_q1, acc_as_e_q1, v_folding_neg[V2 - 1 - cur_dim][0], 2*T_GSW, Q1, Q1_MONT)
    acc_a_q2 = MultiRegisters()
    acc_as_e_q2 = MultiRegisters()
    inv_gadget_ntt_mult_rdim2(q2_twiddles, accumulators[num_per + i], acc_a_q2, acc_as_e_q2, v_folding[V2 - 1 - cur_dim][1], 2*T_GSW, Q2, Q2_MONT)
    print(f"Product2 {test_hash(combine(acc_a_q2))}")
    inv_gadget_ntt_mult_rdim2(q2_twiddles, accumulators[i], acc_a_q2, acc_as_e_q2, v_folding_neg[V2 - 1 - cur_dim][1], 2*T_GSW, Q2, Q2_MONT)
    
    from_ntt_reg(acc_a_q1, acc_a_q2)
    write_regs(accumulators[i].a, acc_a_q1, acc_a_q2)
    from_ntt_reg(acc_as_e_q1, acc_as_e_q2)
    write_regs(accumulators[i].as_e, acc_as_e_q1, acc_as_e_q2)
    print(f"{cur_dim} {i} {test_hash(accumulators[i].a.crt_0.data)}")    

def fold_all_ciphertexts(accumulators: List[Ciphertext], v_folding: List[List[List[CiphertextHalf]]], v_folding_neg: List[List[List[CiphertextHalf]]]):
    num_per = 1 << V2 
    for cur_dim in range(V2):
        num_per = num_per // 2
        for i in range(num_per):
            fold_ciphertexts(accumulators, v_folding, v_folding_neg, num_per, i, cur_dim)

r = "../test_data/"
# r = "./test_data/"

def test_folding():
    test_input = load_polys(r + "client_fold_input.dat", False)
    test_keys = load_polys(r + "client_folding_keys.dat", True, 2*T_GSW, True, True, True)
    test_keys_neg = load_polys(r + "client_folding_keys_neg.dat", True, 2*T_GSW, True, True, True)
    test_output = load_polys(r + "client_fold_output.dat", False)
    # for input in test_input:
    #     print("input-host", test_hash(input.a.crt_0.data))
    fold_all_ciphertexts(test_input, test_keys, test_keys_neg)
    out = test_input[0]
    correct_out = test_output[0]
    if out != correct_out:
        print("Folding incorrect")
    else:
        print("Folding correct")

if __name__ == "__main__":
    test_folding()