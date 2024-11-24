"""
Utilities for working with primes.

Much of this is reimplimentation of sympy facilities.
"""
# Much of this is copied from homework4_notebook.py.
from array import array
from collections.abc import Container, Iterable, Iterator
from math import isqrt

import sympy.ntheory as _sn
# from contextlib import contextmanager, ContextDecorator
# from contextvars import ContextVar

from .util import alternate_impl, USE_SYMPY


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

    def __getitem__(self, item):  # pragma: no cover
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
                    assert False, 'unreachable'  # pragma: no cover
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


def _sn_primes(_s=_sn.Sieve(), /):
    """Generate primes."""
    from itertools import count
    from sympy import sieve
    for i in count(1):
        yield sieve[i]


@alternate_impl(USE_SYMPY, _sn_primes)
def primes(*, _cache=PrimeCache([2, 3, 5, 7, 11, 13])) -> Iterator[int]:
    """Generate primes.

    Caches primes across calls.
    This uses trial division, so it will be inefficient for large primes.

    This method is the basis for its entire module.
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


@alternate_impl(USE_SYMPY, True, _sn.primerange)
def primerange(a: int, b: int | None = None, /) -> Iterator[int]:
    """Generate primes on [2, a) or [a, b).

    See also sympy.ntheory.primerange.
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

    # # Itertools version:
    # from itertools import dropwhile, takewhile
    # yield from takewhile(lambda n: n < b,
    #                      dropwhile(lambda n: n < a, primes()))
    pass


@alternate_impl(USE_SYMPY, _sn.isprime)
def is_prime(n, /):
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


# This sieve is slower than the Itertools recipe, but more space-efficient.
class Sieve:
    """Very space-efficient sieve of Eratosthenes.

    Tracks the primality of the odd numbers up to its size as a bit array of
    half that size.

    Use generate() to get a generator that gets primes from the sieve, filling
    it as needed.

    Use populate() to immediately fill it entirely().

    Indexing by numbers reveals whether they are prime, if they have already
    been generated.
    """
    @property
    def size(self) -> int:
        return self._size

    def __init__(self, size: int, /):
        self._size = size
        odd_bits = size // 2 + size % 2
        array_size = odd_bits // 8 + bool(odd_bits % 8)
        self._data = array('B', b'\xFF' * array_size)
        # Clear the bit representing 1
        self._data[0] &= 0b11111110
        # Clear bits after the end of the needed bits.
        self._data[-1] &= bits.clear_high_bits_mask(-odd_bits % 8)
        # (-x % 8) flips mod 8, but preserves zeros, like ((8 - x % 8) % 8).
        self._full = size < 3 # the smallest sizes are full by default

    def __len__(self):
        return self.size

    @staticmethod
    def _index_of(n, /):
        """Get direct bit index of n.

        Use _index to get the actual position in the internal array.
        """
        if n % 2 == 0:
            raise ValueError('no evens have indices')
        else:
            return (n - 1) // 2

    def _index(self, base: int) -> tuple[int, int]:
        assert base % 2, 'internal index must be odd'
        actual = (base - 1) // 2  # convert integer to odd
        return divmod(actual, 8)

    def _toindex(self, index: int) -> tuple[int, int] | tuple[None, None]:
        if not isinstance(index, int):
            raise TypeError(f'{type(self).__name__} indices must be int,'
                            f' not {type(index).__name__}')
        elif index >= self.size:
            raise IndexError(f'{type(self).__name__} index out of range')

        if index % 2:
            return self._index(index)
        else:
            return None, None

    def __getitem__(self, item: int) -> bool:
        byte, offset = self._toindex(item)
        if item == 2:
            return True
        elif byte is None or offset is None: # check both for mypy
            return False
        else:
            return bool((1 << offset) & self._data[byte])

    def __iter__(self) -> Iterator[bool]:
        data = self._data
        yield from [False, False, True] # 0 1 2
        for i in range(3, self.size):
            if i % 2:
                byte, offset = self._index(i)
                yield bool((1 << offset) & data[byte])
            else:
                yield False

    def populate(self):
        """Fill the sieve immediately.
        """
        if not self._full:
            for p in self.generate():
                pass

    def generate(self) -> Iterator[int]:
        yield 2
        cursor = 3
        while p := self._find_true(cursor):
            # If the sieve was filled asynchronously, stop filling it.
            if not self._full:
                self.setmultiples(p)
            yield p
            cursor = p + 1
        else:
            self._full = True

    # TODO: can this be made more efficient by pregenerating bit masks?
    def setmultiples(self, n, /):
        """Mark all multples of n as non-prime, starting with n**2.
        """
        self._toindex(n)
        data = self._data
        for m in range(n * n, len(self), n + n):
            byte, offset = self._index(m)
            data[byte] &= bits.zero_at(offset)

    def _find_true(self, start: int = 3) -> int | None:
        """Find the first true value not before index i, minimum 3.
        """
        if start < 3:
            raise ValueError("can't search for values before 3"
                             f" in {type(self).__name__}")
        elif start % 2 == 0:
            start += 1
        start, offset = self._index(start)

        # If offset, check the first byte separately
        if offset and start < len(self._data):
            data = self._data[start]
            data &= bits.clear_low_bits_mask(offset)
            if data:
                index = start * 8 + bits.index_lowest_bit(data)
                return 2 * index + 1

        for byte in range(start + bool(offset), len(self._data)):
            if self._data[byte]:
                index = byte * 8 + bits.index_lowest_bit(self._data[byte])
                return 2 * index + 1
        else: # nothing found
            return None


class bits:
    # Namespace for some bit operations
    # Some of the operations will have outputs larger than a byte.

    @classmethod
    def low_bits(bits, byte: int, n: int) -> int:
        return byte & bits.low_bits_mask(n)

    @classmethod
    def low_bits_mask(bits, n: int) -> int:
        return bits.clear_high_bits_mask(8 - n)

    @classmethod
    def high_bits(bits, byte: int, n: int) -> int:
        return byte & bits.high_bits_mask(n)

    @classmethod
    def high_bits_mask(bits, n: int) -> int:
        return bits.clear_low_bits_mask(8 - n)

    @classmethod
    def clear_high_bits(bits, byte: int, n: int) -> int:
        return byte & bits.clear_high_bits_mask(n)

    @staticmethod
    def clear_high_bits_mask(n: int) -> int:
        return 0b11111111 >> n

    @classmethod
    def clear_low_bits(bits, byte: int, n: int) -> int:
        return byte & bits.clear_low_bits_mask(n)

    @staticmethod
    def clear_low_bits_mask(n: int) -> int:
        """May overflow a byte."""
        return 0b11111111 << n

    @staticmethod
    def index_lowest_bit(byte):
        lsb_val = byte & -byte
        return lsb_val.bit_length() - 1

    # @staticmethod
    # def one_at(position):
    #     return 1 << position

    @staticmethod
    def zero_at(position):
        # 1 << position puts a 1 there, and ~ does a Python-bitwise-negation.
        # The resulting number is negative (-position-1), but Python pads
        # negatives with extra ones in bitwise operations.
        return ~(1 << position)

    @staticmethod
    def zero_at2(position):
        """May overflow a byte. Input must be from 0 to 7."""
        return 0b111111101111111 >> (7 - position)

