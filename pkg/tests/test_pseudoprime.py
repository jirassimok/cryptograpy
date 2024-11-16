from collections.abc import Iterable
from dataclasses import dataclass
from itertools import count
from typing import cast, Literal
import unittest

from sympy.ntheory import isprime
from homework.pseudoprime import strong_prime_test

from .test_prime import PRIMES_BELOW_4000


def find_near(pseudoprimes: set[int]) -> tuple[set[int], set[int]]:
    """Find the nearest true prime and composite above and below each
    pseudoprime (excluding the pseudoprimes). Returns a set of primes and a set
    of composites.
    """
    primes = set()
    composites = set()
    for pp in pseudoprimes:
        hi = Near(pp)
        lo = Near(pp)
        for i in count(2, 2):
            # look at numbers 2, 4, ... away from pp
            a = pp + i
            b = pp - i
            if a not in pseudoprimes:
                hi.consider(a)
            if b not in pseudoprimes:
                lo.consider(b)
            if hi.full() and lo.full():
                break
        primes.add(hi.prime)
        primes.add(lo.prime)
        composites.add(hi.composite)
        composites.add(lo.composite)
    return primes, composites


@dataclass
class Near:
    m_prime: int | None = None
    m_composite: int | None = None

    @property
    def prime(self) -> int:
        return cast(int, self.m_prime)

    @property
    def composite(self) -> int:
        return cast(int, self.m_composite)

    def consider(self, x: int):
        prime = isprime(x)
        if self.m_prime is None and prime:
            self.m_prime = x
        elif self.m_composite is None and not prime:
            self.m_composite = x

    def full(self):
        return self.m_prime is not None and self.m_composite is not None


class TestCheckPrime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import homework.util
        homework.util.VERBOSE = False

    def auto_test_bases(self, bases: int | set[int],
                        pseudoprimes: Iterable[int], /):
        pseudoprimes = set(pseudoprimes)
        primes, composites = find_near(pseudoprimes)

        if isinstance(bases, int):
            bases = {bases}

        for p in primes: # true witnesses to primality
            self.assertTrue(strong_prime_test(p, bases), f'p={p}, bs={bases}')
        for c in composites: # true witnesses to compositeness
            self.assertFalse(strong_prime_test(c, bases), f'c={c}, bs={bases}')
        for p in pseudoprimes: # false witnesses to primality
            self.assertTrue(strong_prime_test(p, bases), f'p={p}, bs={bases}')

    # All pseudoprimes below pulled from OEIS (see A001262 for refs)

    def test_base_2(self):
        self.auto_test_bases(2, (2047, 3277, 4033, 4681, 8321, 15841, 29341))

    def test_base_3(self):
        self.auto_test_bases(3, (121, 703, 1891, 3281, 8401, 8911, 10585))

    def test_base_5(self):
        self.auto_test_bases(5, (781, 1541, 5461, 5611, 7813, 13021, 14981))

    def test_bases_2_3(self):
        self.auto_test_bases({2, 3}, [1373653, 1530787, 1987021, 2284453])

    def test_bases_2_5(self):
        self.auto_test_bases({2, 5}, [1907851, 4181921, 4469471, 5256091])

    def test_bases_2_3_5(self):
        self.auto_test_bases({2, 3, 5}, [25326001, 161304001, 960946321])

    def test_bases_2_3_5_7(self):
        self.auto_test_bases({2, 3, 5, 7}, [3215031751, 118670087467])

    def test_4000(self):
        for bases in [(2, 3), (2, 5), (3, 5), (13, 29), (1301, 1871)]:
            for n in range(4000):
                if n in bases:
                    continue
                self.assertEqual(strong_prime_test(n, bases),
                                 n in PRIMES_BELOW_4000,
                                 f'n = {n}, bases = {bases}')
