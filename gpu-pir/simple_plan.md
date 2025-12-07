

Load twiddles using __ldg and hope compiler preloads
Load parameters using __ldg and hope compiler preloads
Read/write input/output to global memory
Shared memory for scratch space for NTTs, 1 per NTT?

Accumulators and NTT live in registers

6, 256-thread blocks with 8KB each with 6 independent queries?

Load db using __ldg since db computed in a separate kernel?

3, 512 thread blocks,
2, 512 thread blocks with more registers (As in 1 1024 thread block)