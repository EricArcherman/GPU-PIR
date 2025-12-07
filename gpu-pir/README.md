# gpu-ntt
Number Theoretic Transform Implementation on GPU for FHE Applications

--------------------

Simon's notes: mul takes a lot longer when it touches all the data - because it's memory bottlenecked
Seems like umulhi is one instruction
Might change with --arch command.
Should inspect ntt barett code

If I use 512 threads because it divides 1536, should I use a radix 4 butterfly?  Probably an optimization for later.  Well I will likely want the Harvey butterfly to avoid the 64 bit operations as well.
The twiddle factor load pattern needs to be investigated.
Constant memory has latency proportional to the number of different addresses.
Texture memory doesn't decrease latency - just bandwidth pressure.
Shared memory write collisions to the same bank serialize.  Fortunately 32 byte bank accesses.
However, the access patterns are mod 32 banks.  So if I have writes 32 apart it's going to suck.
There is also the option of generating them as you go - more mults though.
Would a radix 8 butterfly be better?  Use more registers (which I seem to have plenty of?) and reduces memory accesses?  However, 256 threads.
Use a 2-D thread block - 128 by 2 where the y coordinate specifies the CRT chunk.
Shared memory is definitely needed during NTTs, but that is an 8KB chunk.  The twiddles themselves are 8KB.
Technically you could duplicate the twiddles for each layer and avoid all conflicts.
However, if you want them to be constants then you need to minimize the number of different accesses.
Computing them on demand requires another modular multiply, and may actually be the answer idk.  See https://arxiv.org/pdf/2012.01968.pdf.
Ok so that paper has much bigger things - mine is 32KB that could all fit in CMEM.
Each iteration accesses more elements - O(N) of the twiddles are 1, n/2 are 2, ... all the way up to 2048 values being needed.
So if the NTT algorithm can separate these, you could use CMEM for some and TMEM for others.
Alternatively, the above paper suggests loading them into shared memory, which is different.
You could have an NTT kernel, with the twiddles in shared memory, that loads and computes NTTs.
Then you could have kernels with no shared memory at all, that compute the products with the public parameters, as that will mostly wait on GMEM.
Perhaps the use of 1 twiddle instead of 2 motivated the NTT design.
The warp scheduler will schedule other warps if their is latency.
A warp is 32 threads so there are 16/8 warps to hide the latency of each other.  4 warps run at once.  I suppose the threads slightly offset and then the sync threads brings the warps back together.

Seems like it might end up being memory rather than compute limited though.  So the optimizations might not matter...
Can pipeline queries to buffer public parameter loading and database loading and offset across the nine
On query i
Process query blocks 
Retrieve results from query that ended at i.
Load query section that starts at i+1 and ends at i+num_queries into the space occupied by the one that ended at i.
Load DB block (double buffer) for query i+1 in space from i-1.
So there will be 10 public parameter sections, nine of which are in use and one of which is being written to?
I guess I can read out the results aftey query.
This would require a sync which would make things easier.
Alternatively you could offset by one more and have different streams running in parallel?
The pipelining factor is a function of the data transfer/computation speed if the DB isn't being moved in/out.
Like you could have just 2 buffers but then there would be a lot of downtime?
Does launching a kernel with multiple blocks save that much?  Or each stream is like load->run->get_result

I think is the one to base the design off of:
https://pdfs.semanticscholar.org/e000/fa109f1b2a6a3e52e04462bac4b7d58140c9.pdf/
Has very interesting observation about the needed reductions in a radix-4 transform.
Inconclusive about Shoup vs Montgomery
Flatten the j and k loops.
The number of unique twiddle factors needed is then j ranges from 2^(n-i-1).  so 1024,...8,4,2,1
So the first iteration probably shouldn't be from CMEM, but the last definitely should.
I imagine unrolling the first and last radix-16 iteration.
The first because the input is only in (0,p)
The last to access CMEM, remove the mult where the twiddle is 1, and do whatever is needed for the output.

Actualy then there are only 3 blocks so you're going to code them all anyways.
If it was 2^12 you could do a perfect radix 16?
[1024, 512, 256, 128] TMEM - or maybe you load into into SMEM?
[1, 2, 4, 8]
Stride - writing 16 consecutive elements here - so natively stride 16, might want to offset write orderings - write to base_idx & 0xFFFFFF00 + (i + thread_idx) & 0xFF for i in 0..15.
barrier with SMEM and all values in the same (0, xp)
[16, 32, 64] - reading bank conflicts don't count, write conflict of 2 is probably okay.
[64, 32, 16] CMEM < only 3 here, or maybe 3 on the first iteration>
barrier with SMEM and all values in the same (0, xp)
stride - [128, 256, 512, 1024] -  all elements per thread in the same bank and all threads accessing different banks
[8, 4, 2, 1]  CMEM

I suppose you could also syncronize on the warp instead once if the 32 > 16 values held are those needed for the next butterfly.
But need to know if Amdahl's means this level of optimization is even important

Can profile the basic NTT design with twiddles in SMEM, TMEM, and CMEM easily.  TMEM/CMEM combination a little harder.
Might not see TMEM change since bandwidth isn't saturated by other operations when just running NTT.

GTX 1080 ti : 61
RTX 3090 ti: 86
A100: 80
V100: 70

I think twiddles in SMEM is actually latency optimized and not throughput optimized.
Also the last two layers twiddles are just [+1, -1] and [+1] so you can also just skip CMEM and the modular multiplication.

Option 1:
8KB shared memory, 12 (blocks of 4 warps) * 84SM = 1008 queries.  100 loads
Does NTTs and query with 128 threads
16KB shared memory, 6 (blocks of 8 warps) * 84SM = 504 queries.  200 loads
Does NTTs and query with 256 threads, possibly by doing each CRT with 128 threads in a 2D thread block.
Alternatively could load the twiddle table and input into shared memory.
Less space required for queries, hopefully allowing the whole database (for all queries) to exist in NTT form on GPU.
If not, then that will have to loaded in/out also

Option 2:
Dynamic parallelism:
Parent kenrels have no shared memory, one per query
Each calls child kernels (w/56 way parallelism) to do NTTs.  Might want smaller radix/less registers to leave more room for parent kernels?
Can also batch NTTs with shared memory as in option 3 - e.g a large batches of 14. (and the 1 will be automorphisms when needed)
Needs a larger scratch space to get the NTT results, but could process each faster.

Option 3:
NTT twiddles are in shared memory - might even be faster on its own...
Load things that need to be NTTed into input part of shared memory
For maximum efficiency, one giant kernel with 1536 threads and 100KB shared memory (allocate dynamically for larger than 48KB shared memory)
Has the 8KB twiddle table and uses the rest as input.  Or 4KB twiddles and do barett I guess because 12*8+4=100.
2D thread grid helps manage.  Can also do all the NTTs from the same source binary decomposition to reduce input part loading.
Half of the SMs are assigned each CRT/twiddle table.
NTT and write out

The worst case is if we are latency bound and neither compute nor memory bandwidth bound.
Perhaps it's possible to load a 8KB faster - I hope it doesn't wait for each 128 chunk before issuing the next request.

Example of how to profile:
https://developer.nvidia.com/blog/analysis-driven-optimization-analyzing-and-improving-performance-with-nvidia-nsight-compute-part-2/
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#profile
How to optimize:
https://on-demand.gputechconf.com/gtc/2017/presentation/s7444-Cristoph-Angerer-Optimizing-GPU.pdf

Option 1:
Use 96KB of shared memory to process many queries in parallel and hope computation covers up latency.
Each gets 8KB of shared memory (or 16 for two NTTs) with 128 thread radix-16 and uses it to pass NTT values and twiddles

Option 2:
Use 96KB of shared memory as a giant block to load the same db values for multiple queries or parts of the same query
Load the same public parameter value for both parts by expanding the root's left child and root's right child in parallel
Load the same db value for multiple queries at once

Option 3:
Use 24KB of shared memory as 3 512-thread radix 4 NTT and hope caching is good
Use 40KB of shared memory as 3 512-thread radix 4 NTT and keep twiddles in SMEM
Use 96KB of shared memory as 3 512-thread radix 4 NTT, keep twiddles in SMEM, as well as input 56-bit array and output accumulation text.


Global memory latency is ~500 cycles from figure 14.  Async and sync might be the same for 32 byte elements since it always lines up.
Technically I could load smaller elements and then expand them into shared memory but it may be a pain.  Only if bandwidth limited (trades bandwidth for compute)
Using shared memory for twiddles makes a lot of sense if GMEM or TMEM adds a 500cycle latency.  The computation would be so much faster you could give another kernel the shared memory sooner.  But each kernel has its own overhead that is likely longer than the NTT.
Latency doesn't actually matter if it's covered by computation - I only care about throughput.  Strictly speaking, you could just do everything with global memory and (it would have to be) bandwidth bound.

nvcc -arch=sm_86 -rdc=true -cudart static --machine 64 -use_fast_math -O2 28bit_ntt_test.cu -o 28bit_ntt.cubin -lcudadevrt -std=c++11 --cubin

nvcc -arch=sm_86 -rdc=true -cudart static --machine 64 -use_fast_math -O2 28bit_ntt_test.cu -o 28bit_ntt -lcudadevrt -std=c++11

nvcc -arch=sm_61 -rdc=true -cudart static --machine 64 -use_fast_math -O2 28bit_ntt_test.cu -o 28bit_ntt -lcudadevrt -std=c++11

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#ampere

cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);

This first paper says radix 16 is faster.
Also, it allows delaying the mod, however I only have from 28->32 so at most 4q is ok.
It also notes that too many threads can be a problem
https://arxiv.org/pdf/2003.04510.pdf
Perhaps I simply go to 16 8KB NTTs in parallel.
The other operations will then have some trouble.

A larger radix limits the sync threads operations and allows the global loads to occur earlier since it brings them to the top of the function.
The paper above goes up to radix-32 before register pressure is an issue.
You could conceivably do 64 by reading once, evaluating 32, swapping each output for an input while reading the input and compute the top left half, and then evaluating 32
You can also sync within a warp only for the last levels
You can also try https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-examples-reduction to shuffle values around?

Other cited papers:
https://arxiv.org/pdf/2103.16400.pdf
https://arxiv.org/pdf/2109.14704.pdf
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9251245

https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
Slide 22 indicates using async and breadth first issuing can automatically pipeline queries.
However, I'd like to reuse page-locked memory to copy queries so maybe I could get an event callback?

Issue H2D K D2H for each stream in one thread
then a thread that syncs on each stream
It copies out the result, and copies in the new data when the stream is done and then issues H2D K D2H for that stream again.
Have a lock around the issuing so only one thread at a time does it.
With enough streams, the computation and data transfer should be pipelined.

Divide the scratch/public parameter space into #streams batches
Divide the queries into (space/#streams) batches based on how much space there is to run queries concurrently.

Slides seem to recommend 4 as the number of batches which should be sufficient.

I have 

D2H_1
D2H_2  K_1
D2H_3  K_2  H2D_1
D2H_4  K_3  H2D_2  load_queries_1
D2H_1  K_4  H2D_3  load_queries_2
D2H_2  K_1  H2D_4  load_queries_3
...

I should also be loading from files into normal RAM in parallel.

## File Descriptions:

### helper.h
includes:

- modular power calculation
- reversing bits
- random array creation
- calculating twiddle factors


### ntt_30bit.cuh
includes gpu functions:

- barrett
- ntt
- intt

### 30bit_ntt_test.cu
includes:

the main program for applying ntt, then intt on a randomly generated array


--------------------

## How to run

Compile with: **nvcc -arch=sm_XX -rdc=true -cudart static --machine 64 -use_fast_math -O2 30bit_ntt_test.cu -o 30bit_ntt -lcudadevrt -std=c++11**

(replace **XX** with the compute capability of your GPU)

Run with: **./30bit_ntt**


--------------------


*Coded by: Can Elgezen, Özgün Özerk* \
*Contributed by: Ahmet Can Mert, Erkay Savaş, Erdinç Öztürk*
