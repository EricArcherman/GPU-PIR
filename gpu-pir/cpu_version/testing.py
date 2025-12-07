from params import *
from structures import *
from arith import *
from utils import *

def test_hash(poly):
    if hasattr(poly, "data"):
        poly = poly.data
    # DJB hash
    h = 5381
    for p in poly:
        h = ensure32bit((h << 5) + h + (p % Q1))
    return h

def test_hash_mod(poly: PolyHalf):
    return test_hash([x % Q1 for x in poly.data])

def test_hash_from_mont_mod(poly: PolyHalf):
    return test_hash(from_mont(poly, Q1, Q1_MONT))

def test_hash_regs(r : MultiRegisters, txt):
    p = PolyHalf()
    write_half_regs(p, r)
    print(f"{txt} {test_hash_mod(p)} (Q1)")

def test_hash_reduce(r : List[Registers], txt):
    thread_hashes = [5381 for _ in range(NUM_THREADS)]
    for thread_idx in range(NUM_THREADS):
        h = 5381
        for i in range(POLY_LEN // NUM_THREADS):
            h = ensure32bit((h << 5) + h + r[thread_idx][i])
        thread_hashes[thread_idx] = h
    for o in [16,8,4,2,1]:
        for thread_idx in range(NUM_THREADS):
            if thread_idx & o == 0:
                h_prime = thread_hashes[thread_idx + o]
                h = thread_hashes[thread_idx]
                h = ensure32bit((h << 5) + h + h_prime)
                thread_hashes[thread_idx] = h
    for thread_idx in range(0, NUM_THREADS, 32):
        print(f"({thread_idx}-{txt}) {thread_hashes[thread_idx]}")
        
def to_ciphertexts(polys):
    cts = []
    numct = len(polys) // 2
    for p in range(0, numct):
        cts.append(Ciphertext(polys[p], polys[p+numct]))
    return cts

def to_ciphertext_halfs(cts):
    return ([CiphertextHalf(p.a.crt_0, p.as_e.crt_0) for p in cts],[CiphertextHalf(p.a.crt_1, p.as_e.crt_1) for p in cts])

def to_poly_halfs(ps):
    return ([p.crt0 for p in ps],[p.crt_1 for p in ps])

def group_by(polys, grouping):
    if len(polys) % grouping != 0:
        print("grouping error!")
    return [polys[i:i+grouping] for i in range(0, len(polys), grouping)]

def ntt_to_gpu_form(data, montgomery = False):
    crt0 = PolyHalf()
    crt1 = PolyHalf()
    for i in range(POLY_LEN):
        crt0.data[i] = int.from_bytes(data[i*8: (i+1)*8], 'little')
        crt1.data[i] = int.from_bytes(data[(i + POLY_LEN)*8: (i+POLY_LEN+1)*8], 'little')
        if montgomery:
            # can do with montgomery with R^2 for efficiency
            crt0.data[i] = montgomery_mult(crt0.data[i], Q1_RSQUARED, Q1, Q1_MONT, True)
            crt1.data[i] = montgomery_mult(crt1.data[i], Q2_RSQUARED, Q2, Q2_MONT, True)
    if montgomery:
        crt0 = rearrange(crt0)
        crt1 = rearrange(crt1)
    return Poly(crt0, crt1)

def raw_to_gpu_form(data):
    high = PolyHalf()
    low = PolyHalf()
    for i in range(POLY_LEN):
        v = int.from_bytes(data[i*8: (i+1)*8], 'little')
        high.data[i] = v >> 32
        low.data[i] = v & 0xFFFFFFFF
    return Poly(high, low)

def load_polys(fn, crt_form = True, grouping = 1, as_cts = True, as_halfs=False, montgomery = False):
    with open(fn, mode="rb") as file:
        contents = file.read()
        length = len(contents)
        bytes_per_crt = POLY_LEN * 4
        bytes_per_poly = bytes_per_crt * 2
        polys = []
        if crt_form:
            # inefficient spiral packing
            bytes_per_ntt_poly = bytes_per_poly * 2
            for i in range(0, length, bytes_per_ntt_poly):
                polys.append(ntt_to_gpu_form(contents[i:i+bytes_per_ntt_poly], montgomery))
        else:
            for i in range(0, length, bytes_per_poly):
                polys.append(raw_to_gpu_form(contents[i:i+bytes_per_poly]))
        if grouping != 0:
            if as_cts:
                grouping *= 2
            grouped = group_by(polys, grouping)
            if as_cts:
                for i in range(len(grouped)):
                    grouped[i] = to_ciphertexts(grouped[i])
                    if grouping == 2:
                        grouped[i] = grouped[i][0]
                    if as_halfs:
                        grouped[i] = to_ciphertext_halfs(grouped[i])
            return grouped
        if as_cts:
            polys = to_ciphertexts(polys)
        return polys

if __name__ == "__main__":
    regs = MultiRegisters()
    for thread_idx in range(NUM_THREADS):
        for i in range(POLY_LEN//NUM_THREADS):
            regs.rs[thread_idx][i] = i + thread_idx * (POLY_LEN //NUM_THREADS)
    test_hash_reduce(regs.rs, "test")