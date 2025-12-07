from gadget import *
from utils import *
from sympy import ntt, intt

# Use shared memory?
def automorph(p : PolyHalf, t, mod):
    # __syncthreads() (or barrier wait)
    sm = PolyHalf()
    for i in range(POLY_LEN):
        num = (i * t) // POLY_LEN # >> (LEN_LOG2 - 1)
        rem = (i * t) % POLY_LEN # & (POLY_LEN - 1)
        if num % 2 == 1:
            # 64 bit a[i]
            # or MOD - a[i] in each CRT
            sm.data[rem] = mod - p.data[i]
        else:
            sm.data[rem] = p.data[i]
    # for i in range(POLY_LEN): 
    #     p.data[i] = sm.data[i]
    # barrier arrive
    return sm

    # intt_regs(v_i_a_q1, Q1, Q1_MONT, q1_inv_twiddles)
    # automorph(v_i_a_q1, t, Q1)
    # intt_regs(v_i_a_q2, Q2, Q2_MONT, q2_inv_twiddles)
    # automorph(v_i_a_q2, t, Q2)
    # inv_crt_regs(v_i_a_q1, v_i_a_q2)

def negate_l(sm: PolyHalf, i, mod, negate_at):
    # if (negate_at == 0):
    #     return sm[i]
    # elif (i < POLY_LEN - (1 << negate_at)):
    #     return sm[i + (1 << negate_at)]
    # else:
    #     return mod - sm[i + (1 << negate_at) - POLY_LEN]

    if (negate_at == 0):
        idx = i
    elif (i < POLY_LEN - (1 << negate_at)):
        idx = i + (1 << negate_at)
    else:
        idx = i + (1 << negate_at) - POLY_LEN
    v = sm.data[idx]
    if (negate_at == 0):
        return v
    elif (i < POLY_LEN - (1 << negate_at)):
        return v
    else:
        return mod - v
    

def negate_from_sm(sm:PolyHalf, mod, negate_at):
    r = MultiRegisters()
    for thread_idx in range(NUM_THREADS):
        for (j,z) in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            i = z + thread_idx
            r.rs[thread_idx][j] = negate_l(sm, i, mod, negate_at)
    return r

def automorph_negate_from_sm(sm: PolyHalf, t, t_inv, mod, negate_at):
    r = MultiRegisters()
    for thread_idx in range(NUM_THREADS):
        for (j,z) in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            rem = z + thread_idx
            i = (rem * t_inv) % POLY_LEN # & (POLY_LEN - 1)
            num = (i * t) // POLY_LEN # >> (LEN_LOG2 - 1)
            v = negate_l(sm, i, mod, negate_at)
            if num % 2 == 1:
                v = mod - v
            r.rs[thread_idx][j] = v
    return r

def automorph_full(p1, p2, t, t_inv, big_mod, regs=False):
    r1 = MultiRegisters()
    r2 = MultiRegisters()
    for thread_idx in range(NUM_THREADS):
        for (j,z) in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            rem = z + thread_idx
            i = (rem * t_inv) % POLY_LEN # & (POLY_LEN - 1)
            num = (i * t) // POLY_LEN # >> (LEN_LOG2 - 1)
            v1 = p1.data[i]
            v2 = p2.data[i]
            v = v1 << 32 + v2
            if num % 2 == 1:
                v = big_mod - v
            r1.rs[thread_idx][j] = v >> 32
            r2.rs[thread_idx][j] = v & 0xFFFFFFFF
    if regs:
        return r1, r2
    p1o = PolyHalf()
    p2o = PolyHalf()
    write_half_regs(p1o, r1)
    write_half_regs(p2o, r2)
    return p1o, p2o

# just follow it and get it working
# base port, and optimized port?
def expand_f(v_i: Ciphertext, dst1: Ciphertext, dst2: Ciphertext, i: int, t: int, r: int, params: PublicParameters):
    vi_a_q1 = to_regs(v_i.a.crt_0)
    vi_as_e_q1 = to_regs(v_i.as_e.crt_0)

    vi_a_q2 = to_regs(v_i.a.crt_1)
    vi_as_e_q2 = to_regs(v_i.as_e.crt_1)

    # need to skip ICRT for current automorph setup?  But also INTT constant built in
    from_ntt_reg(vi_a_q1, vi_a_q2)
    from_ntt_reg(vi_as_e_q1, vi_as_e_q2)

    # automorph regs is a mess - write to shared memory and read back
    pi_a_q1 = PolyHalf()
    pi_a_q2 = PolyHalf()
    pi_as_e_q1 = PolyHalf()
    pi_as_e_q2 = PolyHalf()
    write_half_regs(pi_a_q1, vi_a_q1)
    write_half_regs(pi_a_q2, vi_a_q2)
    write_half_regs(pi_as_e_q1, vi_as_e_q1)
    write_half_regs(pi_as_e_q2, vi_as_e_q2)

    # these are in the full modulus not half modulus
    ai_a_q1, ai_a_q2 = automorph_full(pi_a_q1, pi_a_q2, t, t_inv, Q1 * Q2)
    # This one will also be needed in NTT form
    ai_as_e_q1, ai_as_e_q2 = automorph_full(pi_as_e_q1, pi_as_e_q2, Q1 * Q2)
    ntt_ai_as_e_q1 = to_regs(ai_as_e_q1)
    ntt_ai_as_e_q2 = to_regs(ai_as_e_q2)
    ntt_regs(ntt_ai_as_e_q1, Q1, Q1_MONT, q1_twiddles)
    ntt_regs(ntt_ai_as_e_q2, Q2, Q2_MONT, q2_twiddles)
    
    test_hash_from_mont_mod(to_poly(ntt_ai_as_e_q1, ntt_ai_as_e_q2))

    # need NTT form again
    vi_a_q1 = to_regs(v_i.a.crt_0)
    vi_as_e_q1 = to_regs(v_i.as_e.crt_0)

    vi_a_q2 = to_regs(v_i.a.crt_1)
    vi_as_e_q2 = to_regs(v_i.as_e.crt_1)
    # accumulate ntt into vi_as_e
    add_regs(vi_as_e_q1, ntt_ai_as_e_q1)
    add_regs(vi_as_e_q2, ntt_ai_as_e_q2)

    ai = Ciphertext(a=Poly(crt_0=ai_a_q1, crt_1=ai_a_q2), as_e=Poly(crt_0=ai_as_e_q1, crt_1=ai_as_e_q2))
    left = (r != 0) and (i % 2 == 0)
    if left:
        inv_gadget_ntt_mult_rdim1(q1_twiddles, ai, vi_a_q1, vi_as_e_q1, params.left_exp_0[r], T_EXP_LEFT, Q1, Q1_MONT)
    else:
        inv_gadget_ntt_mult_rdim1(q1_twiddles, ai, vi_a_q1, vi_as_e_q1, params.right_exp_0[r], T_EXP_RIGHT, Q1, Q1_MONT)

    if left:
        inv_gadget_ntt_mult_rdim1(q2_twiddles, ai, vi_a_q2, vi_as_e_q2, params.left_exp_1[r], T_EXP_LEFT, Q2, Q2_MONT)
    else:
        inv_gadget_ntt_mult_rdim1(q2_twiddles, ai, vi_a_q2, vi_as_e_q2, params.right_exp_1[r], T_EXP_RIGHT, Q2, Q2_MONT)

    write_half_regs(dst1.a.crt_0, vi_a_q1)
    write_half_regs(dst1.a.crt_1, vi_a_q2)
    write_half_regs(dst1.as_e.crt_0, vi_as_e_q1)
    write_half_regs(dst1.as_e.crt_1, vi_as_e_q2)
    # negate other direction for dst 2?
    # I suppose you can generate the NTT of the polynomial x from the twiddle table.

# negate at beginning of next loop rather than end
# from ntt after gadget and do automorph, shift, and accumulate in raw form
# Is that allowed? or does negation have an interaction with ntt?  Well it should be the other kind of negation
# I think a separate output and a negation bool makes more sense for me
def expand(v_i: Ciphertext, dst: Ciphertext, i: int, t: int, t_inv: int, negate_at: int, last: bool,  params: PublicParameters):
    if (i % 2) == 0:
        p1 = params.left_exp_0[negate_at]
        p2 = params.left_exp_1[negate_at]
        mx = T_EXP_LEFT
    else:
        p1 = params.right_exp_0[negate_at]
        p2 = params.right_exp_1[negate_at]
        mx = T_EXP_RIGHT

    # If I am the right expansion, negate first, otherwise don't!
    # does the negation need to happen in NTT form or is this equivalent?
    # let's just get the 0 path working first
        
    # Maybe the NTT access pattern is better than the automorph
    if (i >> (negate_at)) == 0:
        ia_q1 = to_regs(automorph(v_i.a.crt_0, t, Q1))
        ias_e_q1 = to_regs(automorph(v_i.as_e.crt_0, t, Q1))
        vi_a_q1 = to_regs(v_i.a.crt_0)
        vi_as_e_q1 = to_regs(v_i.as_e.crt_0)
    else:
        ia_q1 = automorph_negate_from_sm(v_i.a.crt_0, t, t_inv, Q1, negate_at)
        ias_e_q1 = automorph_negate_from_sm(v_i.as_e.crt_0, t, t_inv, Q1, negate_at)
        print(i, test_hash(combine(ia_q1)))
        vi_a_q1 = negate_from_sm(v_i.a.crt_0, Q1, negate_at)
        vi_as_e_q1 = negate_from_sm(v_i.as_e.crt_0, Q1, negate_at)
    test_hash_regs(ia_q1, "automorphed input a")
    test_hash_regs(ias_e_q1, "automorphed input as_e")
    test_hash_regs(vi_a_q1, "possibly negated input a")
    test_hash_regs(vi_as_e_q1, "possibly negated input as_e")
    # a goes in to gadget and as_e is added around
    ias_e_q1_ntt = copy_regs(ias_e_q1)
    ntt_regs( ias_e_q1_ntt, Q1, Q1_MONT, q1_twiddles, False)
    test_hash_regs( ias_e_q1_ntt, "ntt automorphed input")
    add_regs(vi_as_e_q1,  ias_e_q1_ntt)
    reduce_regs(vi_as_e_q1, 16, 2, Q1)
    # Q2
    if (i >> (negate_at)) == 0:
        ia_q2 = to_regs(automorph(v_i.a.crt_1, t, Q2))
        ias_e_q2 = to_regs(automorph(v_i.as_e.crt_1, t, Q2))
        vi_a_q2 = to_regs(v_i.a.crt_1)
        vi_as_e_q2 = to_regs(v_i.as_e.crt_1)
    else:
        ia_q2 = automorph_negate_from_sm(v_i.a.crt_1, t, t_inv, Q2, negate_at)
        ias_e_q2 = automorph_negate_from_sm(v_i.as_e.crt_1, t, t_inv, Q2, negate_at)
        vi_a_q2 = negate_from_sm(v_i.a.crt_1, Q2, negate_at)
        vi_as_e_q2 = negate_from_sm(v_i.as_e.crt_1, Q2, negate_at)
    ias_e_q2_ntt = copy_regs(ias_e_q2)
    ntt_regs(ias_e_q2_ntt, Q2, Q2_MONT, q2_twiddles, False)
    add_regs(vi_as_e_q2, ias_e_q2_ntt)
    reduce_regs(vi_as_e_q2, 16, 2, Q2)
    
    inv_crt_regs(ia_q1, ia_q2)

    write_out = Ciphertext()
    # ignore as_e part this is just to reuse the inv_gadget_ntt_mult function
    write_half_regs(write_out.a.crt_0, ia_q1)
    write_half_regs(write_out.a.crt_1, ia_q2)
    
    # Once commented out don't need copy_regs anymore (should compile to noop)
    inv_crt_regs(ias_e_q1, ias_e_q2)
    write_half_regs(write_out.as_e.crt_0, ias_e_q1)
    write_half_regs(write_out.as_e.crt_1, ias_e_q2)

    # Q2
    inv_gadget_ntt_mult_rdim1(q2_twiddles, write_out, vi_a_q2, vi_as_e_q2, p2, mx, Q2, Q2_MONT)
    reduce_regs(vi_a_q2, 2, 1, Q2)
    if not last:
        intt_regs(vi_a_q2, Q2, Q2_MONT, q2_inv_twiddles, False)
    write_half_regs(dst.a.crt_1, vi_a_q2)
    reduce_regs(vi_as_e_q2, 2, 1, Q2)
    if not last:
        intt_regs(vi_as_e_q2, Q2, Q2_MONT, q2_inv_twiddles, False)
    write_half_regs(dst.as_e.crt_1, vi_as_e_q2)
    # Q1
    inv_gadget_ntt_mult_rdim1(q1_twiddles, write_out, vi_a_q1, vi_as_e_q1, p1, mx, Q1, Q1_MONT)
    # no need to icrt
    reduce_regs(vi_a_q1, 2, 1, Q1)
    if not last:
        intt_regs(vi_a_q1, Q1, Q1_MONT, q1_inv_twiddles, False)
    write_half_regs(dst.a.crt_0, vi_a_q1)
    reduce_regs(vi_as_e_q1, 2, 1, Q1)
    if not last:
        intt_regs(vi_as_e_q1, Q1, Q1_MONT, q1_inv_twiddles, False)
    write_half_regs(dst.as_e.crt_0, vi_as_e_q1)

def pre_from(v_i: Ciphertext):
    out = Ciphertext()
    out.a.crt_0.data = intt(v_i.a.crt_0.data, Q1)
    out.a.crt_1.data = intt(v_i.a.crt_1.data, Q2)
    out.as_e.crt_0.data = intt(v_i.as_e.crt_0.data, Q1)
    out.as_e.crt_1.data = intt(v_i.as_e.crt_1.data, Q2)
    return out

def compute_t(r):
    t = (POLY_LEN // (1 << r)) + 1
    t_inv = pow(t, -1, POLY_LEN) # todo: precompute
    return t, t_inv

if __name__ == "__main__":
    input = load_polys("../test_data/client_expand_input.dat")[0] # expansion data - maybe they're helpful but maybe they're broken
    left_keys = load_polys('../test_data/client_expand_left_keys.dat', True, T_EXP_LEFT, True, True, True)
    right_keys = load_polys('../test_data/client_expand_right_keys.dat', True, T_EXP_RIGHT, True, True, True)
    # output = load_polys("../test_data/client_expand_output.dat", True, 1, True, False, True)
    print(f"raw inp: {test_hash(input.a.crt_0)}")
    input = pre_from(input)
    print(f"inv input {test_hash(input.a.crt_0)}")
    e1 = Ciphertext()
    e2 = Ciphertext()
    t, t_inv = compute_t(0)
    l0 = [l[0] for l in left_keys]
    l1 = [l[1] for l in left_keys]
    r0 = [r[0] for r in right_keys]
    r1 = [r[1] for r in right_keys]
    p = PublicParameters(left_exp_0=l0, left_exp_1=l1, right_exp_0=r0, right_exp_1=r1, conversion_0=[], conversion_1=[])
    
    # I see, the output has length 128 because it's also including the regev-gsw ciphertexts with the first dimension expansion
    # assert(e1 == output[0])
    
    # I just want to check the first iteration for now; set last to true
    expand(input, e1, 0, t, t_inv, 0, True, p)
    expand(input, e2, 1, t, t_inv, 0, True, p)
    output = load_polys("../test_data/client_expand_output_1round.dat", True, 1, True, False, True)
    print(len(output))
    print(type(output[0]))
    print(type(e1))
    # Alright so the file has a bunch of zeros, storing 4 byte numbers in 8
    # The file has length
    assert(len(output) == 2)
    assert(e1 == output[0])
    # Right side will require negation in order to match output[1]
    assert(e2 == output[1])
