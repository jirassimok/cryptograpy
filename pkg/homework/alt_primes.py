"""
Utilities of questionable value for working with primes.
"""
import math
from typing import Final, NewType, TypeGuard

import sympy.ntheory as sn

# Use (True)Integral because pyright handles it slightly better.
# (Mypy handles Integer nicely as well, though.)
from .integer_types import Integral as Integer


Prime = NewType('Prime', int)


USE_LIBRARY_PRIME_TEST: Final[bool] = False
USE_LIBRARY_PRIME_GEN: Final[bool] = False and not USE_LIBRARY_PRIME_TEST

if USE_LIBRARY_PRIME_TEST:
    def is_prime(n: int, /) -> TypeGuard[Prime]:
        """Determine if n is prime (or pseudoprime).
        """
        return sn.isprime(n)
elif USE_LIBRARY_PRIME_GEN:
    def is_prime(n_: int, /) -> TypeGuard[Prime]:
        n = Integer(n_)
        if n < 0:
            raise ValueError('Can not check primality of negatives')
        elif n < 2:
            return False
        for p in map(Integer, sn.primerange(math.floor(math.sqrt(n) + 1))):
            # TODO (not really): Why doesn't this type check nicely?
            if p | n:
                return False
        else:
            return True
else:
    # Highly-inefficient primality test3
    def is_prime(n: int, /) -> TypeGuard[Prime]:
        """Determine if n is prime.
        """
        if not isinstance(n, int):
            raise TypeError('is_prime arg must be int')
        elif n < 0:
            raise ValueError('Can not check primality of negatives')
        elif n in (0, 1, 4):
            return False
        elif n in (2, 3, 5, 7):
            return True
        elif not n % 2 or not n % 3:
            return False # multiple of primes above

        # Future primes are all 1 away from a multiple of 6
        for f in range(5, math.floor(math.sqrt(n) + 1), 6):
            c1, c2 = f, f + 2
            if not n % c1 or not n % c2:
                return False
        else:
            return True
