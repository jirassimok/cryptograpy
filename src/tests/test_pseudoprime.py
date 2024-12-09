from collections.abc import Iterable
from dataclasses import dataclass
from itertools import count
from typing import cast, Literal
import unittest

from homework.pseudoprime import strong_prime_test

from . import util


PRIMES_BELOW_4000 = set(util.PRIMES_BELOW_4000)
COMPOSITES_BELOW_4000 = PRIMES_BELOW_4000.symmetric_difference(range(1, 4000))
"""Composite numbers below 4000, and also 1."""


class TestCheckPrime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import homework.util
        homework.util.VERBOSE = False

    # All pseudoprimes below pulled from OEIS (see A001262 for refs)

    def test_base_2(self):
        self.check_base_4000(2, (2047, 3277))
        self.auto_check_bases(2, (4033, 4681, 8321, 15841, 29341),
                              no_sympy='ignore')

    def test_base_3(self):
        self.check_base_4000(3, (121, 703, 1891, 3281))
        self.auto_check_bases(3, (8401, 8911, 10585), no_sympy='ignore')

    def test_base_5(self):
        self.check_base_4000(5, (781, 1541))
        self.auto_check_bases(5, (5461, 5611, 7813, 13021, 14981),
                              no_sympy='ignore')

    def test_bases_2_3(self):
        self.auto_check_bases({2, 3}, [1373653, 1530787, 1987021, 2284453])

    def test_bases_2_5(self):
        self.auto_check_bases({2, 5}, [1907851, 4181921, 4469471, 5256091])

    def test_bases_2_3_5(self):
        self.auto_check_bases({2, 3, 5}, [25326001, 161304001, 960946321])

    def test_bases_2_3_5_7(self):
        self.auto_check_bases({2, 3, 5, 7}, [3215031751, 118670087467])

    def test_4000_multi_base(self):
        """Test the first 4000 positive integers against a few bases.
        """
        for bases in [(2, 3), (2, 5), (3, 5), (13, 29), (1301, 1871)]:
            for n in range(1, 4001):
                if n in bases:
                    continue
                self.assertEqual(strong_prime_test(n, bases),
                                 n in PRIMES_BELOW_4000,
                                 f'n = {n}, bases = {bases}')

    def auto_check_bases(self, bases: int | set[int],
                         pseudoprimes: Iterable[int], /,
                         *, no_sympy: Literal['skip', 'ignore'] = 'skip'):
        """Given a base or set of bases and a list of pseudoprimes, test the
        primality checker.

        Finds the nearest primes and compsites to each of the given
        pseudoprimes (in both directions), and tests those as well as the
        pseudoprimes themselves.

        If sympy is not available, no_sympy governs this function's behavior.
        If no_sympy is 'skip' (the default), skip the test. If no_sympy is
        'ignore', treat the test as passing.
        """
        try:
            import sympy  # noqa: F401
        except ImportError:
            if no_sympy == 'skip':
                self.skipTest('auto_check_bases requires sympy')
            elif no_sympy == 'ignore':
                return
            else:
                raise ValueError('invalid test auto_check_bases(no_sympy)')

        pseudoprimes = set(pseudoprimes)
        primes, composites = find_near(pseudoprimes)

        self.check_bases(bases, pseudoprimes, primes, composites)

    def check_base_4000(self, base: int, pseudoprimes: Iterable[int]):
        """Check a base against the numbers below 4000."""
        pseudoprimes = set(pseudoprimes)
        self.check_bases(base,
                         pseudoprimes,
                         PRIMES_BELOW_4000 - {base},
                         COMPOSITES_BELOW_4000 - pseudoprimes)

    def check_bases(self, bases: int | set[int],
                    pseudoprimes: Iterable[int],
                    primes: Iterable[int],
                    composites: Iterable[int]):
        """Given a base or set of bases, check that the values in each of the
        other arguments are classified as expected.
        """
        if isinstance(bases, int):
            bases = {bases}

        for p in primes: # true witnesses to primality
            self.assertTrue(strong_prime_test(p, bases), f'ps={p}, bs={bases}')
        for c in composites: # true witnesses to compositeness
            self.assertFalse(strong_prime_test(c, bases), f'c={c}, bs={bases}')
        for p in pseudoprimes: # false witnesses to primality
            self.assertTrue(strong_prime_test(p, bases), f'pr={p}, bs={bases}')


def find_near(pseudoprimes: set[int]) -> tuple[set[int], set[int]]:
    """Find the nearest (odd) true prime and composite above and below each
    given pseudoprime (excluding the pseudoprimes). Returns a set of primes and
    a set of composites.

    Note: can't actually check that the returned primes aren't unknown
    pseudoprimes.
    """
    if 2 in pseudoprimes:
        raise ValueError("Can't find primes near 2")
    primes = set()
    composites = set()
    for pp in pseudoprimes:
        hi = Near()
        lo = Near()
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
    """Represents a prime and a composite near another number."""
    m_prime: int | None = None
    m_composite: int | None = None

    @property
    def prime(self) -> int:
        return cast(int, self.m_prime)

    @property
    def composite(self) -> int:
        return cast(int, self.m_composite)

    def consider(self, x: int):
        """"Assign x to the saved prime or composite, as appropraite,
        if not already set.
        """
        from sympy.ntheory import isprime
        prime = isprime(x)
        if self.m_prime is None and prime:
            self.m_prime = x
        elif self.m_composite is None and not prime:
            self.m_composite = x

    def full(self):
        """Check whether both a prime and composite have been set."""
        return self.m_prime is not None and self.m_composite is not None
