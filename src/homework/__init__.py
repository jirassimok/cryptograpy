"""CS789 Project package

This package contains a number of modules.

Core Modules
-------
elgamal  - ElGamal encryption and cracking functions
rsa      - RSA encryption and cracking functions
euclid   - Euclidean algorithm and family
factor   - Fast probabilistic factorization algorithms
fastexp  - Exponentiation by squaring
homework4    - primitive root generation and discrete logarithms
pseudoprime  - Miller-Rabin primality testing
pseudorandom - PRNG implementations

Support Modules
---------------
sieve   - A sieve of Eratosthenes
bitier  - Iterators over bits and the abstract classes for the PRNG classes
util    - Miscellaneous support functions.
          This module includes the VERBOSE constant, which, if set
          to true, will make some of the algorithms in this package
          print intermediate work even when not requested explicitly.
randprime   - Prime generation for initializing other PRNGs
cache_util  - Memoization utilities
bit_class   - A class representing a single bit (primarily for type-checking
              purposes)
"""

__all__ = (
    'elgamal',
    'euclid',
    'factor',
    'fastexp',
    'homework4',
    'pseudoprime',
    'pseudorandom',
    'randprime',
    'rsa',
    # Re-exported functions
    'ElGamal',
    'crack_elgamal',
    'gcd',
    'ext_euclid',
    'pow',
    'find_factor_rho',
    'find_factor_pm1',
    'factors',
    'unique_factors',
    'primitive_root',
    'is_primitive_root',
    'discrete_log',
    'strong_prime_test',
    'is_prime',
    'blum_blum_shub',
    'BlumBlumShub',
    'naor_reingold',
    'NaorReingold',
    'random_prime',
    'random_prime_3mod4',
    'system_random_prime',
    'system_random_prime_3mod4',
)

from . import (
    elgamal,
    euclid,
    factor,
    fastexp,
    homework4,
    pseudoprime,
    pseudorandom,
    randprime,
)

from .elgamal import ElGamal, crack as crack_elgamal
from .euclid import gcd, ext_euclid
from .fastexp import pow
from .factor import (
    find_factor_rho,
    find_factor_pm1,
    factors,
    unique_factors,
)
from .homework4 import primitive_root, is_primitive_root, discrete_log
from .pseudoprime import strong_prime_test, is_prime
from .pseudorandom import (
    blum_blum_shub,
    BlumBlumShub,
    naor_reingold,
    NaorReingold,
)
from .randprime import (
    random_prime,
    random_prime_3mod4,
    system_random_prime,
    system_random_prime_3mod4,
)
