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
def fold_ciphertexts(accumulators: List[Ciphertext], v_folding: List[List[Ciphertext]], v_folding_neg: List[List[Ciphertext]]):
    num_per = 1 << V2
    for cur_dim in range(V2):
        num_per = num_per // 2
        for i in range(num_per):
            # Zero initialized
            acc = Ciphertext()
            inv_gadget_ntt_mult(accumulators[num_per + i], acc, v_folding[V2 - 1 - cur_dim], 2*T_GSW)
            # print(f"Product {test_hash(acc.a.crt_0.data)}")
            # Later: neg in place
            inv_gadget_ntt_mult(accumulators[i], acc, v_folding_neg[V2 - 1 - cur_dim], 2*T_GSW)
            # print(f"Product sum {test_hash(acc.a.crt_0.data)}")
            from_ntt_inplace(acc.a)
            from_ntt_inplace(acc.as_e)
            # print(f"from ntt {(acc.a.crt_0.data[0] << 32) + acc.a.crt_0.data[0]}")
            accumulators[i] = acc
            # print(f"{cur_dim} {i} {test_hash(accumulators[i].a.crt_0.data)}") 

def test_folding():
    test_input = load_polys("client_fold_input.dat", False)
    test_keys = load_polys("client_folding_keys.dat", True, 2*T_GSW, True, True)
    test_keys_neg = load_polys("client_folding_keys_neg.dat", True, 2*T_GSW, True, True)
    test_output = load_polys("client_fold_output.dat", False)

    fold_ciphertexts(test_input, test_keys, test_keys_neg)
    out = test_input[0]
    correct_out = test_output[0]
    if out != correct_out:
        print("Folding incorrect")

if __name__ == "__main__":
    test_folding()