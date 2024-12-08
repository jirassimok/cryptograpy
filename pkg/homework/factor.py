"""Factorization algorithms

Key Functions
-------------
find_factor_rho
find_factor_pm1
"""
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from itertools import count
from math import floor, log

from .euclid import euclid as gcd
from .fastexp import fastexp
from .sieve import Sieve
from .pseudorandom import PRNG
from .util import is_verbose, printer, takebetween, Verbosity


def rho_step(x, n):
    return (fastexp(x, 2, n) + 1) % n


def find_factor_rho(n, *, tries=0):
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
    """
    if n < 5:
        raise ValueError(f"Factorize your small number ({n}) yourself.")
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


def find_factor_pm1(n: int, bound: int, rng: PRNG,
                    *, primes: Sieve | Iterable[int] | None = None,
                    verbose: Verbosity = None):
    """Find a factor of n using Pollard's p-1 algorithm.

    Parameters
    ----------
    n : int
        The number to factor. Must be composite, and not a power of a prime.
    bound : int
        The smoothness bound (aka $B$).
    rng : random.Random
        The random number generator to use in the algorithm.

    Keyword Parameters
    ------------------
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

    def get_exponent(p):
        L = floor(log(n, p))
        print('  (l is ', L, ')', sep='')
        return fastexp(p, L)

    # We use the same p**log_{p}(n) for each b, so cache them.
    exponents = Cache(get_exponent)

    while True:
        b = rng.randrange(1, n)
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


class Cache(defaultdict[int, int]):
    """A defaultdict that passes its missing keys to its default factory.
    """
    def __init__(self, default_factory: Callable[[int], int]):
        super().__init__(default_factory)  # type: ignore[arg-type]

    def __missing__(self, key):
        return self.default_factory(key)


class CachingIterable[T](Iterable[T]):
    """An iterable that caches its values the first time they are seen.
    """
    def __init__(self, base: Iterable[T]):
        self.base = iter(base)
        self.cache: list[T] = []

    def __iter__(self) -> Iterator[T]:
        cache = self.cache
        it = self.base
        i = -1
        while True:
            i += 1
            if i >= len(cache):
                try:
                    new = next(it)
                except StopIteration:
                    return
                cache.append(new)
                yield new
            else:
                yield cache[i]
