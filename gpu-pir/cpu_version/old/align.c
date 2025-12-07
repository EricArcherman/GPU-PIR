#include "stdio.h"
#include "string.h"

int main() {
    int base_idx = 32 * 3;
    // loop order is swapped

    int thread_shifts[32];
    for (int thread_idx = 0; thread_idx < 32; thread_idx++) {
        thread_shifts[thread_idx] = (thread_idx ^ ((thread_idx >> 4) & 0x1)) & 0xF;
    }
    unsigned char touched[16 * 32];
    memset(touched, 0, sizeof(touched));
    for (int i = 0; i < 16; i++) {
        // show what each thread index would write to.
        int thread_access_idxs[32];
        for (int thread_idx = 0; thread_idx < 32; thread_idx++) {
            int start_idx = base_idx + thread_idx * 16;
            // 2 way bank conflict
            // thread_access_idxs[thread_idx] = start_idx + ((thread_idx + i) & 0xF);
            // no bank conflict
            // thread_access_idxs[thread_idx] = start_idx + (((thread_idx ^ i) + (thread_idx >> 4)) & 0xF);
            int idx = start_idx + thread_shifts[thread_idx] ^ i;
            thread_access_idxs[thread_idx] = idx;
            if ((idx < base_idx) || (idx > base_idx + sizeof(touched))) {
                printf("Out of bounds: %d\n", idx);
            } else {
               touched[idx - base_idx] = 1;
            }
        }
        for (int j = 0; j < 32; j++) {
            printf("%03d,", thread_access_idxs[j]);
        }
        printf("\n");
        for (int j = 0; j < 32; j++) {
            printf(" %02d,", thread_access_idxs[j] % 32);
        }
        printf("\n");
    }
    for (int i = 0; i < sizeof(touched); i++) {
        if (touched[i] != 1) {
            printf("Missed index: %d\n", i);
        }
    }
}