"""
Utilities for working with primes.
"""
from collections.abc import Container, Iterable, Iterator
from math import isqrt


__all__ = ['primes', 'primerange', 'is_prime']


# Late addition to the file
class PrimeCache(Container, Iterable[int]):
    """Set of known primes, supporting
    limited operations.

    Assumes all insertions are in ascending
    order without checking.
    """
    @property
    def largest_known(self):
        return self._largest

    def __init__(self, items=(2,)):
        # items is a dict rather than a set because dicts are ordered
        self._items = {i: True for i in items}
        self._largest = max(self._items)

    def __iter__(self):
        return PrimeCacheIterator(self)

    def __contains__(self, item):
        return item in self._items

    def add_largest(self, item):
        self._items[item] = True
        # assert item > self._largest
        self._largest = item

    def __getitem__(self, item):
        if item != -1:
            raise ValueError('Can only retrieve last prime')
        return self._largest


class PrimeCacheIterator(Iterator[int]):
    """An iterator over the prime cache.

    Continues working as expected if the cache grows, but performance will
    suffer. Not thread-safe.
    """
    def __init__(self, cache):
        self._cache = cache
        self._largest = cache.largest_known
        self._it = iter(cache._items)
        self._last = 0

    def __next__(self):
        if self._last is None:
            raise StopIteration
        try:
            if self._largest == self._cache.largest_known:
                p = self._last = next(self._it)
                return p
            else:
                # The cache has grown since the last check.
                # (This special handling isn't actually needed,
                #  because primes() doesn't hold a cache iterator
                #  anywhere it can be interrupted.)
                self._largest = self._cache.largest_known
                self._it = iter(self._cache._items)
                last = self._last
                for p in self._it:
                    if p > last:
                        self._last = p
                        return p
                else:
                    # This only happens if the cache grows incorrectly.
                    assert False, 'unreachable'
        except StopIteration as e:
            self._last = None
            raise e from None


def candidate_primes():
    """Generate candidate primes.

    Generates small primes, then numbers congruent to 5 or 1 mod 6.
    """
    yield 2
    yield 3
    from itertools import count, chain
    yield from chain.from_iterable(
        zip(count(5, 6),
            count(7, 6)))


def primes(*, _cache=PrimeCache([2, 3, 5, 7, 11, 13])) -> Iterator[int]:
    """Generate primes.

    Caches primes across calls.
    This uses trial division, so it will be inefficient for large primes.
    """
    known = _cache
    for c in candidate_primes():
        # skip already-checked numbers
        if known.largest_known >= c:
            if c in known:
                yield c
            continue
        if all(c % p != 0 for p in known):
            # no known prime divides c
            known.add_largest(c)
            yield c


def primerange(a: int, b: int | None = None, /) -> Iterator[int]:
    """Generate primes on [2, a) or [a, b).
    """
    if b is None:
        a, b = 2, a

    it = primes()
    while (p := next(it)) < a:
        continue
    if p < b:
        yield p # first prime >= a
    while (p := next(it)) < b:
        yield p


def is_prime(n, /):
    """Primality test by trial division.
    """
    # return n in primerange(n+1)
    if n < 2:
        return False
    elif n == 2:
        return True
    bound = isqrt(n)
    for p in primerange(bound + 1):
        if n % p == 0:
            return False
    else:
        return True
