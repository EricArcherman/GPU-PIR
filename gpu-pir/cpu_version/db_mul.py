
# multiply database by expansion
# The reorientation is equivalent to doing many small multiplies

# Write DB out in blocks of 4 for coalescing
# Does this mean 4 NTT residues at once per multiply?
# Or is it better to have strided writes and more efficient DB products?
# I guess it depends how much fits in shared memory, which depends on the DB size.
# Also, multi-folding is a thing for large DB entries.



# Hold accumulators in thread registers (so each thread is responsible for some columns)
# Load expanded into shared memory (so each block handles some chunk of rows at a time)
# Cycle through all the rows through SM

# Because we have to load the whole O(n) database and use each element once
# Additional reloading of the O(sqrt(n)) SM may not be incredibly significant
# The whole thing will definitely be memory bottle necked

# Idk if there is an advantage to overlapping with the last part of expansion of first part of folding if that is computaitonally intensive
# E.g. if expansion outputs raw then a ntt could be the first part of fold but that would require full polynomials

from structures import MultiRegisters
from params import *
from arith import *


ACCS_PER_THREAD = 8
ROWS_PER_SM = 2048*512//32  # 3 blocks of 512 threads with 32KB SM each?
DB_STRIDING = 4
# Accumulate on 64 bit numbers, and then reduce?
# Unless it reduces register pressure
ADDS_BEFORE_REDUCE = 8

# Because there's a good chance that adds before reduce is actually 7 and ROWS_PER_SM doesn't divide?
# Well you want it to unroll one and not the other possibly
# CHUNK_LOOP(var_name, end, chunk_size, code) Iterate with chunks of chunk_size and then a truncated chunk if needed
# for(_chunk_iterator = 0; _chunk_iterator < end; _chunk_iterator += chunk_size) { 
#   for(_offset = 0; _offset < chunk_size; _offset++) {
#     var_name = _chunk_iterator + _offset;     
#     #code 
#   }
# }
# for(#var_name = end // chunk_size; #var_name < end) { #code }

def mat_multiply(query, database, threadIdx):
    # I suppose you could try to async load in SM but idk if that's needed
    accs = MultiRegisters()
    col_offset = threadIdx
    reduce_at = ADDS_BEFORE_REDUCE * Q1 # if crt 0 or crt 1
    for i in range(0, 1 << V1, ROWS_PER_SM):
        rows_sm = load_rows(query, i * ROWS_PER_SM, (i+1) * ROWS_PER_SM)
        for row_block in range(0, ROWS_PER_SM, ADDS_BEFORE_REDUCE):
            #pragma unroll
            for offset in range(ADDS_BEFORE_REDUCE):
                row = row_block + offset
                #pragma unroll
                for col in ACCS_PER_THREAD:
                    # 64 bit accumulate
                    accs[2*col : 2*col + 1] += rows_sm[row] * get_db_element(database, row, col + col_offset)
            # can do comparison and subtraction with only high part
            # Honestly this is so efficient idk if it even needs to be delayed, or the inputs fully reduced  
            if accs[2*col : 2*col + 1] >> 32 >= reduce_at:
                accs[2*col : 2*col + 1] -= reduce_at << 32
    montgomery_reduce(accs[2*col : 2*col + 1])
