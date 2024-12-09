"""Factorization algorithms

Key Functions
-------------
find_factor_rho
find_factor_pm1
"""
from collections.abc import Iterable, Iterator
from itertools import count, repeat
from math import floor, log
from random import Random

from .cache_util import Cache, CachingIterable
from .euclid import euclid as gcd
from .fastexp import fastexp
from .sieve import Sieve
from .pseudoprime import is_prime
from .util import is_verbose, printer, takebetween, Verbosity


__all__ = [
    'find_factor_rho',
    'find_factor_pm1',
    'factors',
    'unique_factors',
]


def rho_step(x, n):
    return (fastexp(x, 2, n) + 1) % n


def find_factor_rho(n, *, tries=0, check_prime=False):
    """Find a factor of n using Pollard's rho algorithm.

    Parameters
    ----------
    n : int
        The number to factor. Must be composite and greater than 4.

    Keyword parameters
    ------------------
    tries : int, default 0
        Number of times to try if the algorithm fails. Each try will use a
        larger initial x. If 0, try forever.
    check_prime : bool, default False
        Whether to explicitly check for prime arguments.
    """
    if n < 5:
        if n < 0:
            raise ValueError(f"Can not factor negative number {n}")
        raise ValueError(f"Can not factor small number {n}")
    elif check_prime and is_prime(n):
        raise ValueError(f"Can not factor prime number {n}")

    span = count(2) if tries == 0 else range(2, 2 + tries)
    for x in span:
        y = rho_step(x, n)
        while True:
            g = gcd(abs(x - y), n)
            if 1 < g < n:
                return g
            elif g == 1:
                x = rho_step(x, n)
                y = rho_step(rho_step(y, n), n)
            elif g == n:
                # exit inner loop, go to next initial x
                break
            else:
                assert False, 'unreachable'
    else:
        raise ValueError(f'Failed to find a factor in {tries} tries')


_sieve = Sieve(10_000)


def find_factor_pm1(n: int, bound: int, bases: Random | Iterable[int],
                    *, tries: int = 0,
                    primes: Sieve | Iterable[int] | None = None,
                    verbose: Verbosity = None):
    """Find a factor of n using Pollard's p-1 algorithm.

    Parameters
    ----------
    n : int
        The number to factor. Must be composite, and not a power of a prime.
    bound : int
        The smoothness bound (aka $B$).
    bases : random.Random or iterable of int
        The source to use for the b values in the algorithm. If it is an
        instance of random.Random (including pseudorandom.PRNG), its
        randrange method will be used. Otherwise, it will be used as an
        iterator over b-values to try.

    Keyword Parameters
    ------------------
    tries : int, default 0
        The number of b values to test. If 0 (the default), try forever
        (or until the provided source of b values is exhausted).
    primes : prime.Sieve or iterable of int, optional
        A Sieve of primes containing the factor base, or an iterable that
        will produce consecutive primes starting at 2, at least up to the
        bound. If not given, a default sieve will be used.
    verbose : bool, optional
        If false, print nothing. If true, or if not given and util.VERBOSE
        is true, print some of the intermediate values of the algorithm.
    """
    global _sieve

    print = printer(is_verbose(verbose))

    ## Prepare prime generation

    if primes is None:
        # If the default sieve is too small, replace it with a bigger one.
        if _sieve.size <= bound:
            # round B up to a multiple of 16 for pointless storage optimziation
            _sieve = Sieve(bound + (-bound % 16))
        primes = _sieve
    elif isinstance(primes, Sieve) and primes.size <= bound:
        raise ValueError('provided sieve does not cover the bound')

    # Unify the types of prime providers
    if isinstance(primes, Sieve):
        primes = primes.generate()
    primes = CachingIterable(primes)

    ## Prepare exponents cache

    # We use the same p**log_{p}(n) for each b, so cache them.
    def get_exponent(p):
        L = floor(log(n, p))
        print('  (l is ', L, ')', sep='')
        return fastexp(p, L)
    exponents = Cache(get_exponent)

    ## Prepare b value generation

    # Loop iterator, infinite if 0
    span = repeat(0) if tries == 0 else range(tries)

    if isinstance(bases, Random):
        # Generate b from RNG
        def generate_bs() -> Iterator[int]:
            for _ in span:
                yield bases.randrange(1, n)

    elif bases is None and not isinstance(tries, int):
        # Take b from provided list
        def generate_bs() -> Iterator[int]:
            for b, _ in zip(bases, span):
                yield b

    else:
        raise TypeError('Invalid arguments for p-1 algorithm')

    ## Main algorithm

    for b in generate_bs():
        g = gcd(b, n)
        print('checking', b, '-> gcd is', g)
        if 1 < g < n:
            return g
        elif g == n:
            continue
        # else g == 1
        assert g == 1, '1 <= g <= n is always true'

        for p in takebetween(primes, 1, bound + 1):
            e = exponents[p]
            b = fastexp(b, e, n)
            g = gcd(b - 1, n)
            print('  prime', p, '-> exponent', e, '-> b is', b, '-> gcd is', g)
            if 1 < g < n:
                return g
            elif g == n:
                break  # go to next b
            elif g == 1:
                continue
            else:
                assert False, 'unreachable'
        else:
            # out of primes to check
            assert g == 1, 'we only run out of primes when g is always 1'
            raise ValueError(
                f'{n} has no factors 1 greater than a {bound}-smooth number')
    else:
        raise ValueError('failed to find a factor')


## Useful functions based on these

def factors(n: int) -> Iterator[int]:
    """Generate all prime factors of a number using Pollard's Rho algorithm.

    For an input of 0 or 1, generates no factors.

    Parameters
    ----------
    n : int
        The number to factorize.
    """
    if n == 0:
        return
    elif n == 1:
        return

    while not is_prime(n):
        if n == 4:
            # Special case for tiny n where rho has a bad time
            factor = 2
            yield factor
        elif is_prime(factor := find_factor_rho(n)):
            yield factor
        else:
            yield from factors(factor)
        n //= factor
    else:
        # The final factor
        yield n


def unique_factors(n: int) -> Iterator[int]:
    """Generate unique prime factors of a number using Pollard's Rho algorithm.

    Parameters
    ----------
    n : int
        The number to factorize.
    """
    found = set()
    for factor in factors(n):
        if factor not in found:
            yield factor
            found.add(factor)
