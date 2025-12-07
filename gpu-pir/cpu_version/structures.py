from dataclasses import dataclass, field
from typing import List
from params import *

@dataclass
class PolyHalf:
    data: List[int] = field(default_factory=lambda: [0 for _ in range(POLY_LEN)]) 
    def __eq__(self, obj):
        if not isinstance(obj, PolyHalf) or len(obj.data) != POLY_LEN:
            return False
        for i in range(POLY_LEN):
            if self.data[i] != obj.data[i]:
                return False
        return True

@dataclass
class Poly:
    crt_0: PolyHalf = field(default_factory = PolyHalf)
    crt_1: PolyHalf = field(default_factory = PolyHalf)
    def __eq__(self, obj):
        return isinstance(obj, Poly) and self.crt_0 == obj.crt_0 and self.crt_1 == obj.crt_1

@dataclass
class Ciphertext:
    a: Poly = field(default_factory = Poly)
    as_e: Poly = field(default_factory = Poly)
    def __eq__(self, obj):
        return isinstance(obj, Ciphertext) and self.a == obj.a and self.as_e == obj.as_e

@dataclass
class CiphertextHalf:
    a: PolyHalf = field(default_factory = PolyHalf)
    as_e: PolyHalf = field(default_factory = PolyHalf)
    def __eq__(self, obj):
        return isinstance(obj, CiphertextHalf) and self.a == obj.a and self.as_e == obj.as_e


@dataclass
class PublicParameters:
    left_exp_0: List[List[CiphertextHalf]]
    left_exp_1: List[List[CiphertextHalf]]
    right_exp_0: List[List[CiphertextHalf]]
    right_exp_1: List[List[CiphertextHalf]]
    conversion_0: List[List[CiphertextHalf]]
    conversion_1: List[List[CiphertextHalf]]

@dataclass
class InputForm:
    # A list of bytes
    data: List[int] = field(default_factory=lambda: [0 for _ in range(POLY_LEN // 4)])

@dataclass
class RawForm:
    by_byte: List[InputForm] = field(default_factory=lambda: [InputForm() for _ in range(8)])

@dataclass
class QueryStorage:
    query: Ciphertext
    public_parameters: PublicParameters

    v_folding: List[List[Ciphertext]]
    # May be able to be computed on demand
    v_folding_neg: List[List[Ciphertext]]
    
    accumulators: List[List[Ciphertext]]

@dataclass
class Database:
    data: List[List[List[Poly]]]

# @dataclass
# class  SharedMemory:
#     public_params: List[PolyHalf] = field(default_factory=lambda: [PolyHalf() for _ in range(2)])
#     twiddles: PolyHalf = field(default_factory = PolyHalf)
#     input: List[InputForm] = field(default_factory=lambda: [InputForm for _ in range(EXPANSIONS_PER_BLOCK)]) 
#     # Perhaps a strided list
#     scratch: List[PolyHalf] = field(default_factory = lambda: [PolyHalf() for _ in range(EXPANSIONS_PER_BLOCK)])

# Hopefully a class like this will both
# 1) Use registers instead of local memory
# 2) Inline all the case statements since I can also overload array in c++
@dataclass
class Registers:
    r0: int = 0
    r1: int = 0
    r2: int = 0
    r3: int = 0
    r4: int = 0
    r5: int = 0
    r6: int = 0
    r7: int = 0
    r8: int = 0
    r9: int = 0
    r10: int = 0
    r11: int = 0
    r12: int = 0
    r13: int = 0
    r14: int = 0
    r15: int = 0
    
    def __getitem__(self, key):
        if (NUM_THREADS > 512) and (key >= 2):
            raise IndexError("Only 4 registers")
        if (NUM_THREADS > 256) and (key >= 4):
            raise IndexError("Only 8 registers")
        if (NUM_THREADS > 128) and (key >= 8):
            raise IndexError("Only 16 registers")
        if (key == 0):
            return self.r0
        elif (key == 1):
            return self.r1
        elif (key == 2):
            return self.r2
        elif (key == 3):
            return self.r3
        elif (key == 4):
            return self.r4
        elif (key == 5):
            return self.r5
        elif (key == 6):
            return self.r6
        elif (key == 7):
            return self.r7
        elif (key == 8):
            return self.r8
        elif (key == 9):
            return self.r9
        elif (key == 10):
            return self.r10
        elif (key == 11):
            return self.r11
        elif (key == 12):
            return self.r12
        elif (key == 13):
            return self.r13
        elif (key == 14):
            return self.r14
        elif (key == 15):
            return self.r15
        else:
            raise IndexError("Index beyond 16 or negative")

    def __setitem__(self, key, value):
        if (NUM_THREADS > 512) and (key >= 2):
            raise IndexError("Only 4 registers")
        if (NUM_THREADS > 256) and (key >= 4):
            raise IndexError("Only 8 registers")
        if (NUM_THREADS > 128) and (key >= 8):
            raise IndexError("Only 16 registers")
        if (key == 0):
            self.r0 = value
        elif (key == 1):
            self.r1 = value
        elif (key == 2):
            self.r2 = value
        elif (key == 3):
            self.r3 = value
        elif (key == 4):
            self.r4 = value
        elif (key == 5):
            self.r5 = value
        elif (key == 6):
            self.r6 = value
        elif (key == 7):
            self.r7 = value
        elif (key == 8):
            self.r8 = value
        elif (key == 9):
            self.r9 = value
        elif (key == 10):
            self.r10 = value
        elif (key == 11):
            self.r11 = value
        elif (key == 12):
            self.r12 = value
        elif (key == 13):
            self.r13 = value
        elif (key == 14):
            self.r14 = value
        elif (key == 15):
            self.r15 = value
        else:
            raise IndexError("Index beyond 16 or negative")

@dataclass
class MultiRegisters:
    rs: List[Registers]  = field(default_factory=lambda: [Registers() for _ in range(NUM_THREADS)])

@dataclass
class BufferState:
    twiddle_selected: bool # int if more than 2
    twiddle_token: int
    scratch_token: int 

TWIDDLES_BUFFER = 0
PARAMS_BUFFER = 1
INPUT_BUFFER = 3
SCRATCH_OFFSET = 4
BUFFER_AHEAD = 1