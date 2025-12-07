from sympy import isprime

def check_primes(starting_bits, ntt_degree):
    # Assume that starting_bits > ntt_degree
    # so 2^starting bits + 1 = 1 mod 2*ntt_degree
    p = (1 << starting_bits) + 1
    end = (1 << (starting_bits + 1))
    # maintain candidates that are 1 mod 2 * ntt_degree 
    step = 2*ntt_degree
    m = (1 << 32)
    smallestRemainder = end
    closestPrime = 0
    while p < end:
        if isprime(p):
            # A NTT friendly prime
            remainder = m % p
            negative_remainder = p - remainder
            msbs = (p-1)*(p-1) >> 32 # maximum most significant bits
            if remainder * msbs < m or negative_remainder * msbs < m:
                print(f"{p.bit_length()}: {p} is {p % (2*ntt_degree)} modulo {2*ntt_degree}")
                print(f"2^32 is {remainder} mod {p} and will not overflow")
            r = min(remainder, negative_remainder)
            if r < smallestRemainder:
                smallestRemainder = r
                closestPrime = p
        p += step
    print(f"{starting_bits}: closest: {closestPrime} with distance of {smallestRemainder}")

for b in range(20,32):
    check_primes(b, 2048)

# Needs to be 28 bits for security though
# And can't be more without losing precision.  So stuck with 28 even if something is better
# Let (a * b) = k MSB_24 || l LSB_32.  24 + 32 = 56, the number of bits in a 28 bit product.
# Where LSB_32 is calculated from integer multiplication
# And MSB_24 is the mantessa of the floating point product, shifted by the (exponent - 32)
# Then (a + b) = l mod n + (2^32 mod n) * k mod n
# l mod n with magic number
# k mod n = k
# Want 2^32 mod n small so no overflow
# The two closest options to avoiding overflow in the second modular multiplicaton are
#  286322689 is 126961 away 11110111111110001
#  330391553 is -122893 away
# Let's check the spiral primes
# 268369921 remainder is 11111111111111110000 <- very interesting
# 249561089 remainder is 11000111111111111111101111

# However, having the 56 bit product in two registers is enough to finish Montogmery?
# Let R = 2^32
# Have T mod R, and multiply by N' mod R to get m
# Have T/R.  Need to add mN/R.  I guess another floating point mult extract?