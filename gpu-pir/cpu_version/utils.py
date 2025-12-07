from params import *
from structures import *

# This declaration represents the GPU shared memory as a global variable in python
# SHARED_MEMORY = SharedMemory()

# def copy_to_shared(s: PolyHalf, g: PolyHalf):
#     # to be implemented as async memcpy
#     for i in range(len(s.data)):
#         s.data[i] = g.data[i]

def rearrange(p: PolyHalf):
    out = PolyHalf()
    for thread_idx in range(NUM_THREADS):
        for i, idx in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            z = idx + thread_idx
            out.data[z] = p.data[thread_idx*8 + i]
    return out

def combine(m: MultiRegisters):
    out = PolyHalf()
    for thread_idx in range(NUM_THREADS):
        for j in range(2*NUM_PAIRS):
            out.data[thread_idx*2*NUM_PAIRS + j] = m.rs[thread_idx][j]
    return out.data

def raw_combine(m : MultiRegisters):
    out = PolyHalf()
    for thread_idx in range(NUM_THREADS):
        for i, idx in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            z = idx + thread_idx
            out.data[z] = m.rs[thread_idx][i]
    return out.data

def async_load(s: BufferState, buf_id, needed_val, nxt_val):
    buf_id * BUFFER_AHEAD

def copy_regs(src: MultiRegisters):
    r = MultiRegisters()
    for tid in range (NUM_THREADS):
        for j in range(NUM_REGS):
            r.rs[tid][j] = src.rs[tid][j]
    return r

def to_regs(src: PolyHalf):
    r = MultiRegisters()
    for thread_idx in range (NUM_THREADS):
        for (j,z) in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            r.rs[thread_idx][j] = src.data[z + thread_idx]
    return r

def write_half_regs(dst: PolyHalf, r: MultiRegisters):
    for thread_idx in range(NUM_THREADS):
        for (j,z) in enumerate(range(0, POLY_LEN, NUM_THREADS)):
            dst.data[z + thread_idx] = r.rs[thread_idx][j]

def to_poly(r1: MultiRegisters, r2: MultiRegisters):
    inv_crt_regs(r1, r2)
    
