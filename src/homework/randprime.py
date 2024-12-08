"""Utilities for generating random primes.

"""
from .bititer import PRNG, SystemRandomBitIterator
from .pseudoprime import is_prime


def random_prime(bits: int, rng: PRNG) -> int:
    """Generate a random prime of a given size from the given PRNG.

    Parameters
    ----------
    bits : int
        The number of bits in the prime. The most-significant bit will be 1.
    rng : PRNG
        The RNG to generate the prime from.
    """
    while True:
        p = rng.randrange(2**(bits - 1), 2**bits)
        if is_prime(p):
            return p


def random_prime_3mod4(bits: int, rng: PRNG) -> int:
    """Generate a random prime equal to 3, mod 4, of a given size, from a PRNG.

    Parameters
    ----------
    bits : int
        The number of bits in the prime. The most-significant bit will be 1.
    rng : PRNG
        The RNG to generate the prime from.
    """
    while True:
        p = random_prime(bits, rng)
        if p % 4 == 3:
            return p


def system_random_prime(bits: int) -> int:
    """Generate a random prime from the system's RNG.

    This is useful for getting primes to set up other RNGs.
    """
    return random_prime(bits, SystemRandomBitIterator())


def system_random_prime_3mod4(bits: int) -> int:
    """Generate a random prime equal to 3 mod 4 from the system's RNG.

    This is useful for getting primes to set up other RNGs.
    """
    return random_prime_3mod4(bits, SystemRandomBitIterator())
