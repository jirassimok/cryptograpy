from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from itertools import count
import random

from .integer_types import Bit as _Bit


__all__ = [
    'Bit',
    'asbit',
    'pack_bits',
    'BitIterator',
    'WrappingBitIterator',
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


class RandomBitIterator(BitIterator):
    """A BitIterator based on the system random module.

    This iterator is unsuitable for cryptographic use.
    """
    def __init__(self, seed=None):
        if seed is None:
            self._random = random
        else:
            self._random = random.Random(seed)

    def __next__(self) -> Bit:
        return asbit(self._random.getrandbits(1))

    def next_byte(self) -> int:
        return self._random.getrandbits(8)

    def next_int(self, nbits: int) -> int:
        return self._random.getrandbits(nbits)


class SystemRandomBitIterator(RandomBitIterator):
    """A RandomBitIterator based on random.SystemRandom.
    """
    def __init__(self):
        self._random = random.SystemRandom()
