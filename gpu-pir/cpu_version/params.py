# GPU parameters
# KB - some things don't divide the same and may need to be manually updated
# SHARED_MEMORY_SIZE = 96
# Threads per NTT I guess
NUM_THREADS = 256
# TODO: for testing blockIdx in python code
PY_NUM_BLOCKS = 1
# Set to the current block idx globally
Block_Idx = 0

# Field parameters
POLY_LEN = 2048

NUM_PAIRS = POLY_LEN // NUM_THREADS // 2
NUM_REGS = POLY_LEN // NUM_THREADS
NUM_RADIX = 3
LEN_LOG2 = 11
assert(1 << NUM_RADIX == NUM_PAIRS*2)
assert(1 << LEN_LOG2 == POLY_LEN)
WARP_SIZE = 32

Q1 = 268369921
Q1_ROOT = 66687
Q1_ROOT_INV = 181947619
Q2 = 249561089
Q2_ROOT = 158221
Q2_ROOT_INV = 88293783
# Bezout coefficient * INTT * Montgomery form factor
CRT_INV_FOR1 = (163640210 * pow(POLY_LEN, -1, Q1) * (2**32)) % Q1
CRT_INV_FOR2 = (97389680 * pow(POLY_LEN, -1, Q2) * (2**32)) % Q2
Q1_MONT = 4026597377
Q2_MONT = 4045406209
Q1_RSQUARED = 234877184
Q2_RSQUARED = 148369395
MODULUS = Q1 * Q2
MOD_BITS = 56
PT_MODULUS = 256

# Spiral parameters

# T_CONV = 4
# T_GSW = 7
# T_EXP_LEFT = 16
# T_EXP_RIGHT = 56
# V1 = 10
# V2 = 4

# Fast expansion test params

T_CONV = 4
T_GSW = 8
T_EXP_LEFT = 8
T_EXP_RIGHT = 8
V1 = 6
V2 = 2

# Helper functions for checking accuracy using python
def ensure64bit(x):
    return x & ((1 << 64) - 1)

def ensure32bit(x):
    return x & ((1 << 32) - 1)

if __name__ == "__main__":
    q1_corr = pow(POLY_LEN, -1, Q1)
    q2_corr = pow(POLY_LEN, -1, Q2)
    print(f"Q1 icrt-combined-correction {CRT_INV_FOR1} correction {q1_corr}")
    print(f"Q2 icrt-combined-correction {CRT_INV_FOR2} correction {q2_corr}")
    
RADIX = POLY_LEN // NUM_THREADS