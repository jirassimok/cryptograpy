"""
This module defines functions in two broad groups.

- BitIterator: an iterator over bits
- PRNG: a combination of BitIterator and random.Random.
  The PRNG class is exported by pseudorandom.py rather than this module.

These are used mainly as interfaces for the code implemented
in the pseudorandom module.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from itertools import count
import random

from .bit_class import Bit as _Bit


__all__ = [
    'Bit',
    'asbit',
    'pack_bits',
    'BitIterator',
    'WrappingBitIterator',
    # 'PRNG' is exported from pseudorandom instead
    'WrappingPRNG',
    'RandomBitIterator',
    'SystemRandomBitIterator',
]


type Bit = bool | int | _Bit
"""A type that can represent a bit.

The specific details of this type may change, but these are guaranteed:
- It can be constructed by asbit(int). If the argument is 1 or 0, it will
  have the appropriate value.
- Math between bits and integers works as expected for integers.
- Operations between bits may or may not return bits (this may change in the
  future).
"""


def asbit(b: Bit, /) -> Bit:
    """Convert a value of a bit-compatible type to an actual Bit."""
    if isinstance(b, (_Bit, bool)):
        return b
    elif isinstance(b, int):
        return b % 2
    else:
        return int(b) % 2


def pack_bits(bits: Iterable[Bit]) -> int:
    """Pack bits into an int, from least to most significant."""
    # Uses lowest bits if the inputs aren't really bits.
    return sum(1 << i for bit, i in zip(bits, count()) if bit & 1)


class BitIterator(ABC, Iterator[Bit]):
    """An iterator over bits that can be used to produce bytes or ints.
    """
    @abstractmethod
    def __next__(self) -> Bit:
        ...

    def next_bit(self) -> Bit:
        """Generate a pseudorandom bit."""
        return next(self)

    def next_byte(self) -> int:
        """Generate a pseudorandom byte."""
        return self.next_int(8)

    def next_int(self, nbits: int) -> int:
        """Generate a pseudorandom nbits-bit integer."""
        return pack_bits(next(self) for _ in range(nbits))

    def __iter__(self) -> Iterator[Bit]:
        return self

    def iter_bits(self) -> Iterator[Bit]:
        """Generate bits."""
        return iter(self)

    def iter_bytes(self) -> Iterator[int]:
        """Generate bytes.

        The generator shares this iterator's state.
        """
        yield from self.iter_ints(8)

    def iter_ints(self, nbits: int) -> Iterator[int]:
        """Generate nbits-bit integers.

        The generator shares this iterator's state.
        """
        while True:
            yield self.next_int(nbits)


class WrappingBitIterator(BitIterator):
    """A simple BitIterator that wraps a generator."""
    def __init__(self, generator: Iterator[Bit], /):
        self._generator = generator

    def __next__(self) -> Bit:
        return next(self._generator)


class PRNG(BitIterator, random.Random, ABC):
    """Combination of BitIterator and Python's random.Random.
    """
    def random(self) -> float:
        """Generate a random float on [0, 1)."""
        # float has a 53-bit mantissa
        return self.next_int(53) / (2**53)

    def getrandbits(self, k: int) -> int:
        return self.next_int(k)

    @abstractmethod
    def __next__(self) -> Bit:
        ...

    @abstractmethod
    def seed(self, a=..., version=...) -> None:
        ...

    @abstractmethod
    def getstate(self) -> RngState:  # type: ignore[override]
        ...

    @abstractmethod
    def setstate(self, state: RngState):  # type: ignore[override]
        ...

    def randrange(self: BitIterator, start: int,
                  stop: int | None = None,
                  step: int = 1) -> int:
        """Generate a random int in range(lo, hi).

        The provided bititerator must produce random bits.
        """
        if stop is None:
            start, stop = 0, start
        if stop <= start:
            raise ValueError(f'empty range ({start}, {stop})')

        # count is the number of multiples of step on the interval
        # [0, stop - start] === the number of possible results.
        count = (stop - start + step - 1) // step
        bits = count.bit_length()
        for n in self.iter_ints(bits):
            # find a value from 0 to span-1
            if n < count:
                return start + step * n
        assert False, 'unreachable'


class RngState:
    # Dummy class for RNG state. Not compatible between RNG classes,
    # but parameterizing that wouldn't be worth the benefit.
    pass


class WrappingPRNG(WrappingBitIterator, PRNG):
    """A simple PRNG that wraps a generator."""
    def __init__(self, generator: Iterator[Bit], /):
        super().__init__(generator)

    # The Python docs recommend this for removing overridden methods,
    # though SystemRandom itself raises NotImplementedError instead
    # for the latter two of these methods.
    seed = None  # type: ignore[assignment]
    setstate = None  # type: ignore[assignment]
    getstate = None  # type: ignore[assignment]


class RandomBitIterator(PRNG, random.Random):
    """A BitIterator based on the system random module.

    This iterator is unsuitable for cryptographic use.
    """
    # Has to go back and override the methods from PRNG to
    # prevent recursion.

    def __init__(self, seed=None):
        super().__init__(seed)

    def __next__(self) -> Bit:
        return asbit(self.getrandbits(1))

    def getrandbits(self, k):
        # We can use this super call because PRNG is also a random.Random
        return super(PRNG, self).getrandbits(k)

    def random(self):
        return super(PRNG, self).random()

    def next_int(self, nbits: int) -> int:
        return self.getrandbits(nbits)

    def seed(self, a=None, version=2):
        super(PRNG, self).seed(a, version)

    def getstate(self):
        return super(PRNG, self).getstate()

    def setstate(self, state):
        super(PRNG, self).setstate(state)

    def randrange(self, start, stop=None, step=1):
        return super(PRNG, self).randrange(start, stop, step)


class SystemRandomBitIterator(random.SystemRandom, RandomBitIterator):
    """A RandomBitIterator based on random.SystemRandom.
    """
