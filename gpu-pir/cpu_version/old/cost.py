
# Each NTT is multiplied by 2 public parameters and accumulated into 2 accumulators
# The public parameters are shared between multiple expansions within the query
# For example, expanding the left and right child of the root in parallel
# On the other hand, the accumulators are shared within a gadget product

# This is the total amount of space (Shared memory and registers) required to store all of the inputs to the operation
def exp_poly_cost(NTTs, PublicParams):
    return 3 * NTTs + 2 * PublicParams

# This is how many product terms are computed
def exp_computes(NTTs, PublicParams):
    return 2 * NTTs * PublicParams


a = [[(i, j, exp_poly_cost(i, j), exp_computes(i, j)) for j in range(12)] for i in range(12)]

for row in a:
    print(row)

# I have room for around 36 polynomials if I use the entire shared memory (optimistically)
# I need the number of NTTs to be a power of 2 for my tree trick
# I need the number of NTTs to be a multiple of 3 because of the very goofy 1536 thread thing (which directly conflicts with the above requirement)
# I need the number of public parameters to divide 56, and possibly 7 or 8 as well (56 during expansion and 7 or 8 or 4 during folding and conversion: unknown if other values are possible)

# It seems maybe 8 NTTS with 7 public parameters for 112 total and 38 polynomials could work.  I might be able to do more things if I play tricks with double buffering and pipelining


# To multiply with the database, each query-expanded input is used 4 times, with 4 different database elements, added to 4 different accumulators
# Multiple parts of the same query-expansion add to the same accumulators
# Multiple queries (from different clients) reuse the same database elements
# The number of NTTs above will likely match, or otherwise require reloading of data (am I counting SM cost or data movement cost?)
# Likely some pipeline/double buffering/streaming will be desired

def db_mul_cost(NTTs, queries, accumulators):
    # inputs + db_elements + accumulators
    return NTTs * queries + NTTs * 4 + accumulators * queries * 4

def db_computes(NTTs, queries, accumulators):
    return 4 * NTTs * queries * accumulators

possibilities = []
for i in range(1,12):
    for j in range(1,12):
        for k in range(1,12):
            if db_mul_cost(i,j,k) <= 40:
                possibilities.append((i,j,k,db_mul_cost(i,j,k), db_computes(i,j,k)))

possibilities.sort(key = lambda x: x[4], reverse=True)
print(possibilities[:10])

# Looks like 4 NTT outputs, 1 query, 4 accumulators is a good choice with 36 Polys and 64 multiplies
# Curiously, I could divide the polynomials into their 2048*2 coefficients and do 2048*2 separate multiplies - one for each coefficient.
# We can assume the DB is stored coordinate-wise, and will have to store the NTT output coordinate wise-as well (and the accumulators only exist within the threads)
# It would be lovely then to be able to write 128/4 coordinates at once to write/read 128 bit blocks - which would require 32 simultaneous NTTs in the coefficient expansion.
# (Everything would be reindexed so the 4NTT rows computed together during expansion are then next to each other during the multiplication)
# Otherwise you're only going to be able to write a small number of bytes to each coordinate location.
# This is interesting because I think you pay the full bandwidth overhead.  But on the other hand I could do pairs of 2 or 4 coordinates during the DB multiply and reduce the number of rows/columns in a tile.
# Can also accumulate in a 64bit and then mont? Hmm NTT output would need to be reduced then?  Also mont only works up to 60 bits.  Or a bunch of conditional subtractions and then one mont at the end.

# It could be very nice to have 3 kernels <Idk where conversion goes>
# Coefficient expansion
# DB multiply - then you could change the block size independently.  When A is stored in shared memory, shouldn't it be transposed to avoid bank conflicts?  Or I suppose the DB is stored transposed?
# Folding