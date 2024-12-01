# -*- flycheck-checker: python-mypy; -*-
# Miller-Rabin primality test
from __future__ import annotations
from collections.abc import Iterable

from .fastexp import fastexp as _fastexp
from .util import is_verbose, printer, substr, supstr, Verbosity

# Per Wikipedia, there is no prime below 2**64 that is strongly pseudoprime to
# all these bases:
# 2, 325, 9375, 28178, 450775, 9780504, 1795265022


def fastexp(base: int, exp: int, modulus: int):
    """Modular exponentiation, with special-casing for Fermat's little theorem.

    Identical to fastexp.fastexp(base, exp, modulus), but is never verbose,
    and when that function would return 1 less than the modulus, this one
    returns -1 instead.
    """
    r = _fastexp(base, exp, modulus, verbose=False)
    if r == modulus - 1:
        return -1
    else:
        return r


def strong_prime_test(n: int, bases: Iterable[int], /,
                      *, verbose: Verbosity = None):
    """Miller-Rabin primality test.

    Parameters
    ----------
    n : int
        The number to test.
    bases : iterable of int
        The bases to test against.

    Keyword Parameters
    ------------------
    verbose : bool, optional
        If false, print nothing. If true, or if not given and util.VERBOSE
        is true, print the arguments, the steps of each base's test, and the
        results for each tested base.
    """
    print = printer(is_verbose(verbose))

    if n < 2:
        return False
    elif n == 2:
        return True
    elif n % 2 == 0:
        return False

    m = n - 1
    r = 0
    while m % 2 == 0:
        r += 1
        m //= 2

    print(f'n = {n} = 2^{r} * {m}, bases={bases}')
    for b in bases:
        if not _test_prime_base(n, m=m, r=r, b=b, verbose=verbose):
            print('not prime to base', b)
            return False
        print('prime to base', b)
    else:
        return True


def _test_prime_base(n: int, /, *, r: int, m: int, b: int,
                     verbose: Verbosity = None) -> bool:
    """Partial Miller-Rabin primality test.

    Parameters
    ----------
    n : int
        The number to test.
    r : int
        The number of times 2 divides n - 1.
    m : int
        The largest factor of n - 1 with no even factors (i.e. (n-1)/(2**r)).
    b : int
        The base to test.
    verbose : bool, optional
        If false, print nothing. If true, or if not given and util.VERBOSE
        is true, print the arguments, and the steps of the test.
    """
    print = printer(is_verbose(verbose))

    bs = fastexp(b, m, n)
    print(f'b₀ = {bs} = {b}{supstr(m)} % {n}')

    if bs == 1 or bs == -1:
        return True

    for s in range(1, r):
        last = bs
        bs = fastexp(bs, 2, n)
        print(s)
        print(f'b{substr(s)} = {bs} = {last}² % {n}')
        if bs == -1:
            return True
        elif bs == 1:
            return False
    else:
        return False


def is_prime(n: int) -> bool:
    """Check probabilistically if a number is prime.

    Is accurate for all n up to 318665857834031151167460.
    """
    if n % 2 == 0 or n % 3 == 0 or n < 2:
        return False

    # Reference:
    # OEIS sequence A014233, pseudoprimes relative to first n primes as bases
    # Also referenced: https://doi.org/10.1090/S0025-5718-2014-02830-5

    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)

    if n < 2047:
        return strong_prime_test(n, primes[:1])
    elif n < 1373653:
        return strong_prime_test(n, primes[:2])
    elif n < 25326001:
        return strong_prime_test(n, primes[:3])
    elif n < 3215031751:
        return strong_prime_test(n, primes[:4])
    elif n < 2152302898747:
        return strong_prime_test(n, primes[:5])
    elif n < 3474749660383:
        return strong_prime_test(n, primes[:6])
    elif n < 341550071728321:  # same for 7-8 prime bases
        return strong_prime_test(n, primes[:7])
    elif n < 3825123056546413051:  # same for 9-11 prime bases
        return strong_prime_test(n, primes[:9])
    elif n < 318665857834031151167461:
        return strong_prime_test(n, primes[:12])
    else: # elif n < 3317044064679887385961981:
        return strong_prime_test(n, primes)
