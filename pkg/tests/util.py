from collections.abc import Sequence
from itertools import islice

from homework.pseudorandom import PRNG, split_bits
from homework.bititer import WrappingPRNG

type BitLength = int


class FalseWrappingPRNG(WrappingPRNG):
    """Lying PRNG that just produces one value at a time from certain methods.
    """
    def randrange(self, start, stop=None):
        return next(self._generator)

    def randint(self, start, stop=None):
        return next(self._generator)


def false_random(values: Sequence[int | tuple[int, BitLength]]) -> PRNG:
    """Create a PRNG that will produce the given values at the given sizes.

    Numbers without sizes are generated as-is.

    Note: if these numbers will be used for randrange, use FalseWrappingPRNG
    instead (or in addition).
    """
    def generator():
        for val, size in values:
            yield from islice(split_bits(val), size)

    return WrappingPRNG(generator())
