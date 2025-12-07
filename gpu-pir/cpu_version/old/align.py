
#  Compute 3x mod 2047 to shuffle indexes between banks
from ntt import bitReverse


# for i in range(2048):
#     x = 3 * i
#     y = (x >> 11) + x & 2047
#     if (y >= 2047):
#         print(f"{i} {y}")
# # The only one that would need a second iteration is also the one that would be a problem because it would conflict with 0
# print(3 * 2047)
# print((3*2047)>> 11)
# print((3*2047)&2047)

# Hmm the effect only applies when you roll over the modulus
# We can try to make it apply within the warp
# for i in range(0,2048,16):
#     x = 3 * (i & 511)
#     y = ((x >> 9) + (x & 511)) ^ (i & (1024+512))
#     # print(f"{i} {y} {y%32}")

# # Checking bit reverse
# for i in range(0,2048,16):
#     y = bitReverse(i, 11)
#     # print(f"{i} {y} {y%32}")

# # Checking bit reverse
# for i in range(0,2048,16):
#     x = 3 * bitReverse(i, 11)
#     y = (x >> 11) + x & 2047
#     # print(f"{i} {y} {y%32}")

# done = False
# for k in range(1,100000):
#     if done:
#         break
#     for i in range(2048):
#         x = k * i
#         y = (x >> 11) + x & 2047
#         z = (y >> 11) + y & 2047
#         if (i == 2047):
#             if (z != 2047):
#                 print(f"{k} {i} {z}")
#                 done = True
#         elif (z >= 2047) and (i % 23 != 0) and (i % 89 != 0):
#             print(f"{k} {i} {z}")
#             done = True
#             break

# Larger is better?
# for i in range(2048):
#     x = 2045 * i
#     y = (x >> 11) + x & 2047
#     z = (y >> 11) + y & 2047
#     if (y >= 2047):
#         print(f"{i} {z}")

# No that's just multiplication by -2
# for i in range(0,2048,16):
#     x = 2045 * i
#     y = (x >> 11) + x & 2047
#     z = (y >> 11) + y & 2047
#     print(f"{i} {z} {z % 32}")

def is_valid(k):
    for i in range(2048):
        x = k * i
        y = (x >> 11) + x & 2047
        z = (y >> 11) + y & 2047
        if (z >= 2047) and y != 2047:
            return False
        elif (y == 2047) and z != 2047:
            return False
    return True

def f(i, k):
    x = k * i
    y = (x >> 11) + x & 2047
    z = (y >> 11) + y & 2047
    return z

def resolves_conflicts(k):
    conflicts = 0
    for warp in range(4):
        for j in range(16):
            accesses = [False for _ in range(32)]
            for thread in range(warp*32, (warp+1)*32):
                base_idx = thread*16
                i = f(base_idx+j, k) % 32
                if accesses[i]:
                    conflicts += 1
                accesses[i] = True
    return conflicts

min_conflicts = 1000
for k in range(2048):
    if is_valid(k):
        c = resolves_conflicts(k)
        if c < min_conflicts:
            min_conflicts = c
            print(f"{k} {c}")
        elif c == 0:
            print(f"{k} {c}")

def write_conflicts(k):
    conflicts = 0
    for warp in range(4):
        for j in range(8):
            accesses = [False for _ in range(32)]
            for thread in range(warp*32, (warp+1)*32):
                idx = (thread >> 4) * 64 + thread % 16 + j*16
                i = f(idx, k) % 32
                if accesses[i]:
                    conflicts += 1
                accesses[i] = True
    return conflicts

for k in range(0, 2048, 128):
    print(f"{k} {write_conflicts(k)}")

# I wonder if moving the first 2 bits to the back or bit reversing would help

def count_conflicts(f):
    conflicts = 0
    for warp in range(4):
        for j in range(16):
            accesses = [0 for _ in range(32)]
            for thread in range(warp*32, (warp+1)*32):
                base_idx = thread*16
                i = f(base_idx+j) % 32
                accesses[i] += 1
            conflicts += max(accesses) - 1
    for warp in range(4):
        for j in range(8):
            accesses = [0 for _ in range(32)]
            for thread in range(warp*32, (warp+1)*32):
                base_idx = 16 * j + thread * 16 * 8 + ((thread >> 7) & 1)
                i = f(base_idx+j) % 32
                accesses[i] += 1
            conflicts += max(accesses) - 1
        for j in range(8):
            accesses = [0 for _ in range(32)]
            for thread in range(warp*32, (warp+1)*32):
                base_idx = 1024 + 16 * j + thread *16 * 8 + ((thread >> 7) & 1)
                i = f(base_idx+j) % 32
                accesses[i] += 1
            conflicts += max(accesses) - 1
    return conflicts

print(f"{count_conflicts(lambda x: f(x, 128))}")
# add a few read conflicts for less write conflicts
print(f"{count_conflicts(lambda x: f(x, 126))}")

print(f"{count_conflicts(lambda x: f(bitReverse(x, 11), 126))}")

min_conflicts = 1000
for k in range(2048):
    c = count_conflicts(lambda x: f(x, k))
    if c < min_conflicts:
        min_conflicts = c
        print(f"{k} {c}")
    c = count_conflicts(lambda x: f(bitReverse(x, 11), k))
    if c < min_conflicts:
        min_conflicts = c
        print(f"reverse {k} {c}")

# Multiplies, compares, min/max, are also the same number of cycles
# So you could conceivably do an even more complicated function