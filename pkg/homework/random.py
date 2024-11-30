"""
Key features of this module:

- blum_blum_shub: simple, readable implementation of that algorithm
- BlumBlumShub: more complex implementation (implements random.Random)
- naor_reingolg: somewhat simple, readable implementation of that algorithm
- NaorReingoldGenerator: more-complex implementation of that algorithm
- NaorReingoldRandom: wrapper around the other implementation to
  add the random.Random interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import count, starmap
from operator import mul
from random import Random as PyRandom
from typing import cast, overload, Self

from .euclid import _silent_euclid as gcd
from .fastexp import _silent_fastexp as fastexp
from .util import (asbit, Bit, BitIterator, printer, Verbosity, is_verbose,
                   WrappingBitIterator)


# This alias just makes some of the other APIs clearer.
type PRNG = BitIterator


def dot(v: Iterable[Bit], u: Iterable[Bit]) -> Bit:
    return asbit(sum(starmap(mul, zip(v, u))))


def pad_bits(n: int, bits: int) -> int:
    """Zero-pad n to the given number of bits.

    Actually doesn't pad at all; we rely on Python's arbitrary-size
    integer semantics, which already zero-pad non-negative numbers.
    """
    if n < 0:
        raise ValueError('can not bit-pad negative number')
    elif n.bit_length() > bits:
        raise ValueError(
            f'can not pad {n.bit_length()}-bit number to {bits} bits')
    return n


def split_bits(x: int) -> Iterator[Bit]:
    """Get the bits of a non-negative int, from least to most significant.

    This generator will never end; it will left-pad with infinite-zeroes.
    """
    # Actually, this works perfectly well on negative numbers. I just want
    # to detect any time a negative number somehow gets here.
    if x < 0:
        raise ValueError("Can't split negative numbers")
    while True:
        yield x % 2
        x >>= 1


def randrange(rng: PRNG, lo: int, hi: int, /) -> int:
    """Generate a random int in range(lo, hi)."""
    span = hi - lo
    bits = span.bit_length()
    for n in rng.iter_ints(bits):
        # find a value from 0 to span-1
        if n < span:
            return lo + n
    assert False, 'unreachable'


def random01(rng: PRNG) -> float:
    """Generate a random float on [0, 1)."""
    # float has a 53-bit mantissa
    return rng.next_int(53) / (2**53)


def randint(rng: PRNG, lo: int, hi: int, /) -> int:
    """Generate a random int from lo to hi, inclusive."""
    return randrange(rng, lo, hi + 1)


class LowBitIterator(WrappingBitIterator):
    """WrappingBitIterator that specifically truncates its bits.

    This allows it to work as expected even if the underlying generator isn't
    actually producing only a single bit at a time.
    """
    def __init__(self, generator: Iterator[Bit] | Iterator[int], /):
        super().__init__(asbit(b % 2) for b in generator)


class HybridRandom(BitIterator, PyRandom, ABC):
    # Combination of BitIterator and Python's random.Random.
    def random(self) -> float:
        return random01(self)

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


class RngState:
    pass


## Blum-Blum-Shub


# This is a simple, readable implementation of BBS. The one afterwards is
# more complicated, because it has to expose its internal state.
def blum_blum_shub(p: int, q: int) -> Callable[[int], PRNG]:
    """Create a Blum-Blum-Shub PRNG.

    Parameters
    ----------
    p : integer
        A large prime equal to 3, mod 4.
    q : integer
        The second large prime equal to 3, mod 4.

    Returns
    -------
    callable
        A function that takes in a seed and returns a BitIterator over the
        pseudorandom bits.
    """
    if p % 4 != 3:
        raise ValueError('p % 4 != 3')
    elif q % 4 != 3:
        raise ValueError('q % 4 != 3')
    n = p * q
    del p, q

    def generator(s: int) -> Iterator[Bit]:
        while True:
            s = fastexp(s, 2, n)
            yield asbit(s % 2)

    return lambda seed: WrappingBitIterator(generator(seed))


class BlumBlumShub(HybridRandom):
    """Blum-Blum-Shub PRNG.

    Can be constructed using either n or p and q.

    Attributes
    ----------
    modulus : int
        The modulus for the generator.
    check_seeds : bool, default True
        Whether to check if seeds are coprime to n. Can be overridden when
        the seed is given.

    Parameters
    ----------
    n : int, optional
        The modulus for the generator. Must not be given with p or q.
        If n is not given, p and q must be given.
    p : int, optional
        A large prime number equal to 3, mod 4. Must be given with q.
    q : int, optional
        A large prime number equal to 3, mod 4. Must be given with p.
    seed : int, optional
        The initial seed for the RNG. If not given, the default is a value
        with the same bit length as n.

    Examples
    --------
    >>> p, q = 127, 311
    >>> n = p * q
    >>> BlumBlumShub(n)
    >>> BlumBlumShub(p, q)
    >>> BlumBlumShub(n=n)
    >>> BlumBlumShub(p=p, q=q)
    """
    # Redundant positional / keyword-only overloads are needed because the type
    # checkers have issues when the names are different for maybe-positional
    # parameters. (The first two overloads should just be (self, /, n: int).)
    #
    # Might've been a better idea to only allow (p, q), and not (n).
    @overload
    def __init__(self, n: int, /, *,
                 seed: int | None = ..., check_seeds: bool = ...): ...

    @overload
    def __init__(self, /, *, n: int,
                 seed: int | None = ..., check_seeds: bool = ...): ...

    @overload
    def __init__(self, p: int, /, q: int,
                 *, seed: int | None = ..., check_seeds: bool = ...): ...

    @overload
    def __init__(self, /, *, p: int, q: int,
                 seed: int | None = ..., check_seeds: bool = ...): ...

    def __init__(self,
                 n_or_p: int | None = None,
                 q_or_none: int | None = None,
                 /, *,
                 n: int | None = None,
                 p: int | None = None,
                 q: int | None = None,
                 seed: int | None = None,
                 check_seeds: bool = True):
        # Normalize n/p/q to only n
        q = q if q_or_none is None else q_or_none
        if q is None:
            if p is not None:
                raise TypeError('p; no q')
            n = n if n_or_p is None else n_or_p
            if n is None:
                raise TypeError('missing argument')
        elif n is not None:
            raise TypeError('got n and q')
        else:
            p = p if n_or_p is None else n_or_p
            if p is None:
                raise TypeError('q; no p')
            if p % 4 != 3:
                raise ValueError('p % 4 != 3')
            elif q % 4 != 3:
                raise ValueError('q % 4 != 3')
            else:
                n = p * q
        del p, q, n_or_p, q_or_none

        if seed is None:
            seed = 1 << n.bit_length()

        self._modulus = n
        self._state = seed
        self.check_seeds = check_seeds

    @property
    def modulus(self):
        return self._modulus

    def __next__(self) -> Bit:
        self._state = fastexp(self._state, 2, self.modulus)
        return asbit(self._state & 1)

    def seed(self, a=None, version=2) -> None:
        if not isinstance(a, int):
            raise TypeError(
                f'{type(self).__name__} does not support non-int seeds')
        elif version != 2:
            raise ValueError(f'unsupported seed version: {version}')
        check_seed = self.check_seeds
        if check_seed and gcd(self.modulus, a) != 1:
            raise ValueError('illegal seed')

        self._state = a % self.modulus

    def getstate(self) -> RngState:  # type: ignore[override]
        return cast(RngState, self._state)

    def setstate(self, state: RngState):  # type: ignore[override]
        self._state = cast(int, state)


## Naor-Reingold

def naor_reingold(nbits, p, q,
                  pairs: Iterable[tuple[int, int]],
                  square_root,
                  r) -> PRNG:
    """Iterate over random numbers using a Naor-Reingold function.

    Parameters
    ----------
    nbits : int
        Number of bits.
    p : int
        An nbits-bit prime.
    q : int
        Another nbits-bit prime.
    pairs : iterable of pairs of ints
        The 2*nbits random numbers from 1 to p*q to use in the algorithm.
        They must already be paired up.
    square_root : int
        A number coprime to p*q.
    r : sequence of ints
        A sequence of nbits random bits.
    """
    if p.bit_length() != nbits:
        raise ValueError(f'{p} does not have {nbits} bits')
    elif q.bit_length() != nbits:
        raise ValueError(f'{q} does not have {nbits} bits')
    n = p * q
    del p, q

    pairs = tuple(pairs)
    for a, b in pairs:
        if not (1 <= a <= n):
            raise ValueError(f'number {a} is not on [1, {n}]')
        if not (1 <= b <= n):
            raise ValueError(f'number {b} is not on [1, {n}]')
    del a, b

    if gcd(square_root, n) != 1:
        raise ValueError('square_root not coprime to p*q')
    square = fastexp(square_root, 2, n)
    del square_root

    r = tuple(r)
    if len(r) != nbits:
        raise ValueError('r is not nbits bits long')

    def f(x) -> Bit:
        """The Naor-Reingold function."""
        exp = sum(a[bit] for a, bit in zip(pairs, split_bits(x)))
        g_to_e = fastexp(square, exp, n)
        return dot(r, split_bits(g_to_e))

    return WrappingBitIterator(map(f, count()))


# TODO: Rename parameter/attribute 'r'.
class NaorReingoldGenerator(BitIterator):
    """Naor-Reingold PRNG.

    Can be constructed from all of its parameters via its normal constructor,
    or from just the bit-size, the primes, and a (P)RNG, using from_rng.

    Attributes
    ----------
    nbits : int
        Number of bits. (Called $n$ in the lecture notes.)
    n : int
        The product of two primes. (Called $N$ in the lecture notes.)
    pairs : sequence of pairs of ints
        Random numbers used in the algorithm. (Called $a_{i,j}$ in the lecture
        notes.)
    square : int
        An integer that is square in the integers mod n. (Called $g$ in the
        lecture notes.)
    r : sequence of ints
        A list of nbits bits. (Called $r$ in the lecture notes.)

    Parameters
    ----------
    nbits : int
        Number of bits.
    p : int
        An nbits-bit prime.
    q : int
        Another nbits-bit prime.
    pairs : iterable of pairs of ints or iterator of ints
        The 2*nbits random numbers from 1 to p*q to use in the algorithm. They
        may be provided either as an iterable of pairs of ints or a flat
        iterator (such as another PRNG).

        If an iterator is provided, excess values will not be examined.

        Note: the pairs may not be provided as an iterator of pairs.
    square_root : int
        A number coprime to p*q.
    r : sequence of ints
        A sequence of nbits random bits.
    """
    # Has one additional private variable:
    # _count : iterator of int
    #     The internal generator for arguments to the Naor-Reingold function.

    def f(self, x, *, verbose: Verbosity = None) -> Bit:
        """The Naor-Reingold function."""
        print = printer(is_verbose(verbose))
        exp = sum(a[bit] for a, bit in zip(self.pairs, split_bits(x)))
        g_to_e = fastexp(self.square, exp, self.n)
        print('exp', exp)
        print('g^exp', g_to_e)
        return dot(self.r, split_bits(g_to_e))

    # Function that uses another PRNG to generate most of the parameters.
    @classmethod
    def from_rng(cls, nbits: int, p: int, q: int,
                 rng: PRNG | Iterator[Bit] | Iterator[int]) -> Self:
        """Prepare a Naor-Reingold PRNG using another PRNG.

        Parameters
        ----------
        nbits : int
            Bit-size parameter.
        p : int
            An nbits-bit prime.
        q : int
            Another nbits-bit prime.
        rng : PRNG or iterator of ints or iterator of bits
            A source of random bits to initialize the algorithm. If a non-bit
            iterator is given, only the low bit of each int will be used.
        """
        if not isinstance(rng, BitIterator):
            # We'll assume any bititerator passed here was a PRNG.
            rng = LowBitIterator(rng)

        n = p * q

        def repeat_randint(lo, hi):
            while True:
                yield randint(rng, lo, hi)

        pairs = repeat_randint(1, n)

        while (square_root := randrange(rng, 1, n)):
            if gcd(square_root, n) == 1:
                break

        r = [rng.next_bit() for _ in range(nbits)]

        return cls(nbits, p, q, pairs, square_root, r)

    def __init__(self, nbits: int,
                 p: int,
                 q: int,
                 pairs: Iterator[int] | Iterable[tuple[int, int]],
                 square_root: int,
                 r: Sequence[Bit]):
        if p.bit_length() != nbits:
            raise ValueError(f'{p} does not have {nbits} bits')
        elif q.bit_length() != nbits:
            raise ValueError(f'{q} does not have {nbits} bits')
        n = p * q

        if isinstance(pairs, Iterator):
            pairs = tuple((next(pairs), next(pairs)) for _ in range(nbits))
        else:
            pairs = tuple(pairs)
        for a, b in pairs:
            if not (1 <= a <= n):
                raise ValueError(f'number {a} is not on [1, {n}]')
            if not (1 <= b <= n):
                raise ValueError(f'number {b} is not on [1, {n}]')

        if gcd(square_root, n) != 1:
            raise ValueError('square_root not coprime to p*q')

        r = tuple(r)
        if len(r) != nbits:
            raise ValueError('r is not nbits bits long')

        self._nbits = nbits
        self._n = n
        self._pairs = pairs
        self._square = fastexp(square_root, 2, n)
        self._r = r
        self._count = count()

    @property
    def nbits(self) -> int:
        return self._nbits

    @property
    def n(self) -> int:
        return self._n

    @property
    def pairs(self) -> Sequence[tuple[int, int]]:
        return self._pairs

    @property
    def square(self) -> int:
        return self._square

    @property
    def r(self) -> Sequence[Bit]:
        return self._r

    def __next__(self):
        return self.f(next(self._count))


class NaorReingoldRandom(PyRandom):
    # Just read NaorReingoldGenerator; this is 95% just processing parameters
    # for that class.
    """An implementation of Python's random.Random using NaorReingoldGenerator.

    Does not support re-seeding the RNG.

    Has the same constructor as NaorReingold, including the from_rng factory.
    """
    @classmethod
    def from_rng(cls, nbits: int, p: int, q: int,
                 rng: PRNG | Iterator[Bit] | Iterator[int]) -> Self:
        return cls(NaorReingoldRandom.from_rng(nbits, p, q, rng))

    @overload
    def __init__(self, nbits: int, /, p: int, q: int,
                 pairs: Iterator[int] | Iterable[tuple[int, int]],
                 square_root: int, r: Sequence[Bit]): ...

    @overload
    def __init__(self, /, *, nbits: int, p: int, q: int,
                 pairs: Iterator[int] | Iterable[tuple[int, int]],
                 square_root: int, r: Sequence[Bit]): ...

    @overload
    def __init__(self, rng: NaorReingoldRandom, /): ...

    @overload
    def __init__(self, /, *, rng: NaorReingoldRandom): ...

    def __init__(self, nbits_or_rng: int | NaorReingoldRandom | None = None, /,
                 p: int | None = None,
                 q: int | None = None,
                 pairs: (Iterator[int] | Iterable[tuple[int, int]]
                         | None) = None,
                 square_root: int | None = None,
                 r: Sequence[Bit] | None = None,
                 nbits: int | None = None,
                 rng: NaorReingoldRandom | None = None):
        if nbits is None and rng is None and nbits_or_rng is None:
            raise TypeError('must provide NaorReingold instance or args')
        elif isinstance(nbits_or_rng, int):
            if nbits is not None:
                raise TypeError('extra nbits parameter')
            nbits = nbits_or_rng
        elif isinstance(nbits_or_rng, NaorReingoldRandom):
            if rng is not None:
                raise TypeError('extra rng parameter')
            rng = nbits_or_rng
        else:
            raise TypeError("expected int or NaorReingold,"
                            f" but got '{type(nbits_or_rng)}'")

        # assert (nbits is None) ^ (rng is None)

        if nbits is not None:
            if (p is None or q is None or pairs is None
                or square_root is None or r is None):  # noqa:E129
                raise TypeError('missing arguments')
            rng = NaorReingoldRandom(nbits, p, q, pairs, square_root, r)
        elif (p is not None or q is not None or pairs is not None
              or square_root is not None or r is not None):
            raise TypeError('got both rng and rng constructor args')
        else:
            pass # just given rng

        self._rng = rng

    def getstate(self):
        next_x = next(self._rng._count)
        self.setstate(next_x)  # so we don't skip it
        return next_x

    def setstate(self, state):
        self._rng._count = count(state)

    seed = None  # type: ignore  # pyright: ignore

    def random(self):
        return random01(self._rng)

    def getrandbits(self, k):
        return self._rng.next_int(k)

