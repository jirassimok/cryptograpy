from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TypedDict
import unittest

from homework.bititer import PRNG
import homework.pseudorandom as pr


def skip(it: Iterator, n: int):
    for _ in range(n):
        next(it)


NR_EXAMPLE_ARGS: NRArgs = {
    'nbits': 6,
    'p': 37,
    'q': 47,
    # Reverse pairs because I operate from least- to most-significant
    'pairs': tuple(reversed([(129, 978), (1350, 71), (3, 1028),
                             (514, 526), (411, 495), (216, 810)])),
    'square_root': 359,
    'r': tuple(reversed([1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1])),
}


class TestNaorReingoldClass(unittest.TestCase):
    def test_examples(self) -> None:
        """Test the examples from the lecture.
        """
        # This test is much more compelling as an indicator of the algorithm's
        # correctness when manually comparing the verbose outputs of
        # rng.f(verbose=True) to the calculations we did by hand.
        rng = pr.NaorReingold(**NR_EXAMPLE_ARGS)
        self.assertEqual(rng.f(43), 1, 'f(43)')
        self.assertEqual(rng.f(1), 1, 'f(1)')
        self.assertEqual(rng.f(2), 0, 'f(2)')

    def test_factory(self) -> None:
        rng1 = pr.NaorReingold(**NR_EXAMPLE_ARGS)
        pr.NaorReingold.from_rng(10, 661, 941, rng1)


class TestNaorReingoldFunction(unittest.TestCase):
    def test_examples(self) -> None:
        """Test the examples from the lecture.
        """
        rng = pr.naor_reingold(**NR_EXAMPLE_ARGS)
        next(rng)  # 0
        self.assertEqual(next(rng), 1, 'f(1)')
        self.assertEqual(next(rng), 0, 'f(2)')
        skip(rng, 40)
        self.assertEqual(next(rng), 1, 'f(43)')


class AbstractBBSTest[T](ABC, unittest.TestCase):
    @staticmethod
    @abstractmethod
    def prep_blum_blum_shub(p: int, q: int, /) -> T:
        ...

    @staticmethod
    @abstractmethod
    def seed(bbs: T, seed: int, /) -> PRNG:
        ...

    def blum_blum_shub(self, p: int, q: int, seed: int, /) -> PRNG:
        return self.seed(self.prep_blum_blum_shub(p, q), seed)

    def test_errs(self) -> None:
        p, q = 19, 23

        # Bad choice of prime
        bad = 17  # not 3 mod 4
        with self.assertRaisesRegex(ValueError, '% 4'):
            self.prep_blum_blum_shub(p, bad)
        with self.assertRaisesRegex(ValueError, '% 4'):
            self.prep_blum_blum_shub(bad, q)

        with self.assertRaisesRegex(ValueError, 'prime'):
            self.prep_blum_blum_shub(p, 18)
        with self.assertRaisesRegex(ValueError, 'prime'):
            self.prep_blum_blum_shub(18, q)

        # Bad choice of seed
        rng = self.prep_blum_blum_shub(p, q)
        with self.assertRaisesRegex(ValueError, 'illegal seed'):
            self.seed(rng, 2 * p)
        with self.assertRaisesRegex(ValueError, 'illegal seed'):
            self.seed(rng, 0)

    def assertSquares(self, bbs: T, n: int, seed: int, squares: Sequence[int]):
        rng = self.seed(bbs, seed)
        for i, (expect, actual) in enumerate(zip(squares, rng)):
            self.assertEqual((expect % n) & 1, actual,
                             f'squares of {squares[0]}, index {i}')

    def test_simple(self):
        p, q = 19, 23  # p * q = n = 437
        n = p * q
        # repeated squares of 2, 3, and 5
        s2 = [2, 4, 16, 256, 65536, 4294967296, 18446744073709551616]
        s3 = [3, 9, 81, 6561, 43046721, 1853020188851841]
        s5 = [5, 25, 625, 390625, 152587890625, 23283064365386962890625]
        s6 = [6, 36, 1296, 1679616, 2821109907456, 7958661109946400884391936]
        s233 = [233, 54289, 2947295521, 8686550888106661441,
                75456166331666628614079195878996196481]

        bbs = self.prep_blum_blum_shub(p, q)
        self.assertSquares(bbs, n, 2, s2)
        self.assertSquares(bbs, n, 3, s3)
        self.assertSquares(bbs, n, 5, s5)
        self.assertSquares(bbs, n, 6, s6)
        self.assertSquares(bbs, n, 233, s233)


type SeedFn = Callable[[int], PRNG]


class TestBlumBlumShubFunction(AbstractBBSTest[SeedFn]):
    @staticmethod
    def prep_blum_blum_shub(p: int, q: int) -> SeedFn:
        return pr.blum_blum_shub(p, q)

    @staticmethod
    def seed(bbs: SeedFn, seed: int) -> PRNG:
        return bbs(seed)


class TestBlumBlumShubClass(AbstractBBSTest[pr.BlumBlumShub]):
    @staticmethod
    def prep_blum_blum_shub(p: int, q: int) -> pr.BlumBlumShub:
        return pr.BlumBlumShub(p, q)

    @staticmethod
    def seed(bbs: pr.BlumBlumShub, seed: int) -> PRNG:
        bbs.seed(seed)
        return bbs

    def blum_blum_shub(self, p: int, q: int, seed: int) -> PRNG:
        return pr.BlumBlumShub(p, q, seed=seed)

    def test_errs(self):
        super().test_errs()
        p, q = 19, 23
        # Also check in the constructor
        with self.assertRaisesRegex(ValueError, 'illegal seed'):
            self.blum_blum_shub(p, q, 2 * p)
        with self.assertRaisesRegex(ValueError, 'illegal seed'):
            self.blum_blum_shub(p, q, 0)


del AbstractBBSTest


## Typing helpers

class NRArgs(TypedDict):
    nbits: int
    p: int
    q: int
    pairs: Iterable[tuple[int, int]]
    square_root: int
    r: Sequence[int]
