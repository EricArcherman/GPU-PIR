
# Computation
Baseline:    5ms for 100k NTT on 3090 Ti.   22ms on 1080
Montgomery:  4ms for 100k NTT on 3090 Ti.   16ms on 1080
512 threads: 3ms for 100k NTT on 3090 Ti.   14ms on 1080

3090 has 1536 threads per SM - 512 should improve occupancy
1080 has 2048 threads per SM - so 512 makes occupancy no better


nvcc -arch=sm_61 -cudart static --machine 64 -use_fast_math -O2 test_test.cu host.cu -o test -lcudadevrt -std=c++11

nvcc -arch=sm_86 -cudart static --machine 64 -use_fast_math -O2 test_test.cu host.cu -o test -lcudadevrt -std=c++11

ssh slangows@128.30.93.84
nvcc -arch=sm_86 -cudart static --machine 64 -use_fast_math -O2 28bit_ntt_test.cu host.cu -o 28bit_ntt -lcudadevrt -std=c++11
nvcc -arch=sm_86 -cudart static --machine 64 -use_fast_math -O2 28bit_ntt_test.cu -o 28bit_ntt.cubin -lcudadevrt -std=c++11 --cubin

nvdisasm 28bit_ntt.cubin > 28bit_ntt.ptx
nvdisasm -cfg test.cubin | dot -ocfg.png -Tpng


/usr/local/cuda-11/bin/ncu -o profile --set full 28bit_ntt
/usr/local/cuda-11/bin/ncu -o test --set full test
scp slangows@128.30.93.84:/home/slangows/gpu-pir/profile.ncu-rep ~/c/gpu-pir/profile.ncu-rep
scp slangows@128.30.93.84:/home/slangows/gpu-pir/profile.ncu-rep ~/c/gpu-pir/test.ncu-rep
ncu-ui profile.ncu-rep

256 threads: 3ms for 100k
128 threads: 3ms for 100k
Seems like the dynamic allocation just allocates more if needed.
128 threads 100KB shared memory: 3ms for 100k
64 threads - which shouldn't fit: 3ms for 100k
Hmm so is it not half occupancy?  I'm missing something...
32 threads - which shouldn't fit: 8ms
128 threads explicit twiddle loading, caching disabled: 3ms for 100k

INTT is now a bit faster that the scaling has been moved out of it.
If I'm bandwidth bound then I won't see any speedup
I did see ~10% speedup so I'm not XD

Note that loading both CRT and inverse CRT tables i 32KB which exceeds the remaining 28KB of cache.  Can either use N/2 twiddles or use that inverse values are negation of forward values.
After radix-16 though, decide which is inverse and has to do other lookup.  Subtraction can just be build in to the operations

A radix-16 NTT using Montgomery multiplication, 128-threads.
Twiddles in L1 and nothing else allowed in L1.
4-4-3 perhaps, idk.  Maybe 4-3-4.  Last twiddle is 1 and many corrections can be removed
swap INTT and NTT (it already seems to be faster and should have less corrections in radix-16)
Can take input as 16 registers and output 16 registers?.
NTT function is not inlined, but unrolled within
All other calling functions are unrolled/inlined with parameters for loops/masking


# Movement of data on GPU
Batched by 4 within a query.  First 4 are pre-expanded.  512-threads per query.  3 queries in parallel on a SM.
Store public parameters in 7KB and expand, stored by load order.
Loads same public parameter row for all 4 expansions.
Loads many db elements (sparsely).
Loads accumulator and uses for all 4 parts of the query.
Store bit decompositions in 2048 bit chunks, stored by load order.
Intermediate products in (0, 16p) in 8KB.

Batch folding by 4 - corresponding to default spiral n of 2.

–dlcm=cg to disable L1 thrashing and then use __ldg for twiddle loading

Can make so you only need one of the two elements of the 56 bit number by having 7 bit masks write 28 bits to each

# Movement of data to GPU
Pipeline data movement and computation
Divide queries into 4 batches, and use 4 asynchronous streams.

D2H_1
D2H_2  K_1
D2H_3  K_2  H2D_1
D2H_4  K_3  H2D_2  load_queries_1
D2H_1  K_4  H2D_3  load_queries_2
D2H_2  K_1  H2D_4  load_queries_3

# DB loading
Transfer DB entries in plaintext form to GPU (because smaller)
convert (to Montgomery) and NTT on GPU.
Store in global memory.

# Other
Above assumes computation or bandwidth limited.
It could still be too long, and you need to load pre-expanded queries.
It could also be really fast, and over-optimization is like 50 seconds to 45 seconds.

# Public parameters
Store parameters and RNG seed as sent for smaller files/faster file reading
Expand RNG on CPU (or GPU if you're really up for the task)
Store Montgomery multiplied parameters
So either RNG then multiply or have the client RNG and divide so the RNG is the multiplied value

# Division of shared memory
4 by x
Store accumulators in registers - requires 56-64KB
Store NTT values in registers - requires 32KB
Load public parameters - requires 28-32KB
Load input in corresponding chunks - requires 1KB (1bit) 4 * 2 = 8KB (8bit)
Twiddles - 7-8KB

NTT scratch - requires 8KB-32KB,34KB if strided
Acc scratch - always less than NTT scratch

Potential configurations:
8.6 has
256KB registers
96KB shared memory

1x1024
64 registers per thread
256KB of registers
96KB of shared memory
^ Might even be able to do 8 by x.  However, folding is by 4.
Expansion can also compress input more especially if you compress multiple queries together.
There actually is a factor of 3 if you consider looking at 8+4+2+1 as a stream since the parameters are the same regardless of level.

3x512
42 registers per thread (have to round down)
84KB of registers
32KB of shared memory

Also, on 4090 or datacenter (some)

2x1024 (for other)
32 registers per thread
128KB of registers
80KB of shared memory


I think 4 by x 1 block of 1024 is the simplest design
I keep acc_a as_e in registers (2*8 registers)
I compute NTT (8 registers)
This leaves 40 extra registers for compiler, and 8 if I have 2 by 2024
The shared memory can be divided as
| 8KB twiddles, 32 KB public parameters, 16KB input, 34KB NTT scratch |
No additional synchronization in NTT required - just async read the next when done with the last
No acc communication required - accumulators stay completely within thread
In fact, you have enough registers to store the twiddles in registers to if you really want. (8 registers)

To reuse scratch:

Write sm
sync
Read sm
async_arrive       async_wait
                   write sm
                   sync
                   read sm

Write sm
sync
Read sm
sync_warp
async memcpy

A 3x512 implementation would require scratch reuse
It would have to use cooperative groups to synchronize between blocks (assigning work based on the SM-ID)
Then it would have to do L1-prefetching to load shared data

256 + 100 KB of fast memory

A NTT-ACC requires
8KB for NTT values
14KB for ACC values
2KB for input
24KB minimum.

Overall requires
14KB for 2 public parameters
8KB for twiddles (perhaps in L1, although technically the public parameters could be L1 also)

Then we have

4 NTT: 256 threads in a 1024 block:    96KB, 1/4xbandwidth, 2xc  128KB
6 NTT: 256 threads in 3 x 512 blocks: 144KB, 1/2xbandwidth, 2xc  192KB
8 NTT: 128 threads in a 1024 block:   192KB, 1/8xbandwidth, 1xc  256KB
12NTT: 128 threads in 3 x 512 blocks: 288KB, 1/4xbandwidth, 1xc  384KB single scratch (so requires scratch reuse), 288KB also overflows registers (so part of SM would have to be dedicated to storing values, leaving very few registers anywhere)

So all should technically be possible.
Can implement 4.
If bandwidth bound, then try 8
If latency bound, then try 6
If throughput bound, then improve NTT

4 x 1 (double buffer) and 4 x 2 take the same amount of public parameter space and should have the same computation-latency covering
Loading the input in scratch space after use allows input size to vary easier; it still has a loading latency to deal with
4 x 1 takes half the scratch space which makes things much easier.
But if the input-load latency is now smaller than the 256-NTT (and not the 128-NTT) there will now be a hang.
But you could do 4 x 2 as 2 consecutive 256-NTTs using the same scratch space (can't load input there then)
I suppose there's a case for one scratch space, a 1024-NTT, and it just keeps reusing it on different inputs
All of SM is filled with inputs, parameters
Twiddles in registers
Accumulators somewhere

6 NTT:
8 KB scratch, used twice by 512-thread NTT, 2x4 registers, 2x8 acc
16 KB public params
8 KB as 2x4KB inputs
Twiddles in L1/const
3 blocks

4 NTT:
8 KB scratch, used 4x by 1024-thread NTT - synchronization blocks whole processor versus multiple NTTs in block.s
16 KB public params
16KB as 4x4KB input
Twiddles in scratch?
Double buffer params in scratch?
