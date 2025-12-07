#include <stdio.h>

// return the number of consecutive zeros
int __clz(unsigned int x) {
    if (x == 0) {
        return 32;
    } else {
        return __builtin_clz(x);
    }
}

// compute the next expansion/folding
unsigned int compute(unsigned int previous, unsigned int direction) {
    return (1 << previous) + direction;
}

// do something with dfs output
void output(unsigned int* cache, int idx_log_2) {
    for (int i = 0; i <= idx_log_2; i++) {
        printf("%u, ", cache[i]);
    }
    printf("\n");
}

int dfs_iter(int idx_log_2, int query) {
    int last_idx = 0;
    unsigned int cache[idx_log_2+1];
    cache[0] = query;
    for (int i = 1; i <= idx_log_2; i++) {
        // init left
        cache[i] = 0; //compute(cache[i - 1], 0);
    }
    output(cache, idx_log_2);
    for (int idx = 1; idx < (1 << idx_log_2); idx++) {
        int overlap = 32 - __clz(idx ^ (idx - 1));
        // printf("%d: %d\n", idx, overlap);
        for (int c = idx_log_2 - overlap + 1; c <= idx_log_2; c++) {
            // compute c according to bit i
            int bit = idx & (1 << c);
            cache[c] = idx; //compute(cache[c - 1], bit);
        }
        output(cache, idx_log_2);
    }
}

int main() {
    dfs_iter(9, 5);
}