# mypy: check-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import count
from math import prod
import unittest

import homework.factor as ff
import homework.pseudorandom as pr
from homework.pseudoprime import is_prime

from .test_prime import PRIMES_BELOW_4000


# Note: this class is deleted at the end of the file
# so it isn't run as a test class itself.
class FactorizeTestCase(ABC, unittest.TestCase):
    @abstractmethod
    def find_factor(self, n: int) -> int:
        ...

    def checkPrimes(self, *primes: int):
        for p in primes:
            if not is_prime(p):
                self.fail(f"Bad test case: {p} not prime")

    def assertIsFactor(self, n: int, factor: int, msg=None, /):
        if msg is None:
            msg = 'Not a factor'
        else:
            msg = f'Not a factor [{msg}]'
        self.assertTrue(n // factor * factor == n, msg)

    def test_general(self) -> None:
        self.assertIsFactor(n := 6, self.find_factor(n))
        self.assertIsFactor(n := 10, self.find_factor(n))
        self.assertIsFactor(n := 20, self.find_factor(n))
        self.assertIsFactor(n := 24, self.find_factor(n))
        self.assertIsFactor(n := 11 * 13, self.find_factor(n))
        self.assertIsFactor(n := 900, self.find_factor(n))
        self.assertIsFactor(n := 37 * 43 * 47, self.find_factor(n))

    def test_powers_of_two(self) -> None:
        for x in range(3, 32):
            self.assertIsFactor(n := 2**x, self.find_factor(n), f'2**{x}')


class TestRho(FactorizeTestCase):
    def find_factor(self, n):
        return ff.find_factor_rho(n)

    def assertFindsFactor(self, n, msg=None):
        self.assertIsFactor(n, self.find_factor(n), msg)

    def test_larger_primes(self) -> None:
        # 32-bit primes
        p = 3474614551
        q = 3355591627
        self.assertFindsFactor(p * q)
        r = 3014268083
        s = 2868607237
        t = 3975481921
        self.assertFindsFactor(r * s * t)

    def test_big_numbers(self):
        # This is a very large number with only small prime factors
        n = prod(PRIMES_BELOW_4000[4:])  # 5635-bit number
        self.assertEqual(ff.find_factor_rho(n), 8891699, 'many low primes')

        ps = [63761, 51631, 49429, 50893, 46511]
        self.checkPrimes(*ps)
        self.assertFindsFactor(prod(ps), '16-bit primes')

        ps = [1073016809, 550944223, 799456643, 926260649]
        self.checkPrimes(*ps)
        self.assertFindsFactor(prod(ps), '30-bit primes')


class TestPMinus1(FactorizeTestCase):
    """Test Pollard's p-1 algorithm with two techniques

    1. By using a seeded PRNG, we can choose values where the algorithm always
       finds (or doesn't find) a factor.
    2. By feeding the algorithm a list of b values instead of a PRNG, we can
       make its behavior even more predictable.
    """
    @classmethod
    def setUpClass(cls):
        # Pre-populate the default prime sieve for the algorithm
        from homework.sieve import Sieve
        ff._sieve = Sieve(10_000_000)
        ff._sieve.populate()

    def find_factor(self, n):
        # Instead of randoms, use consecutive increasing numbers
        return ff.find_factor_pm1(n, 100, range(2, n - 2))

    def assertFindsFactor(self, n: int, bound, bases, *, msg=None):
        factor = ff.find_factor_pm1(n, bound, bases)
        self.assertIsFactor(n, factor, msg)

    def test_examples(self):
        # These are the examples from the lecture notes
        self.assertEqual(ff.find_factor_pm1(9991, 3, [3]), 97)
        self.assertEqual(ff.find_factor_pm1(3801911, 5, [3]), 1801)

    def test_low_smoothness(self) -> None:
        # Primes with p-1 of known, small smoothness
        f29 = 1 + 2 * 3 * 5 * 11 * 17 * 29  # 18-bit
        f31 = 1 + 2 * 13 * 23 * 31          # 15-bit
        f43a = 1 + 2 * 17 * 19 * 37 * 43    # 20-bit
        f43b = 1 + 2 * 17 * 19 *      43    # 15-bit  # noqa: E222
        f57 = 1 + 2 * 11**2 * 19 * 37 * 57  # 24-bit
        self.checkPrimes(f29, f31, f43a, f43b, f57)

        rng = self.rng(9876543**3)

        # Check with the minimum smoothness needed
        self.assertFindsFactor(f29 * f31, 29, rng, msg='29 by 31')
        self.assertFindsFactor(f43a * f43b, 43, rng, msg='43 by 43')
        self.assertFindsFactor(f43a * f57, 43, rng, msg='43 by 57')
        self.assertFindsFactor(f31 * f43a * f43b, 31, rng,
                               msg='31 by 43 by 43')
        self.assertFindsFactor(f43a * f43b * f57, 43, rng,
                               msg='43 by 43 by 57')
        self.assertFindsFactor(f29 * f31 * f43a * f57, 29, rng,
                               msg='29 by 31 by 43 by 57')

        # Check with large smoothness bounds
        self.assertFindsFactor(
            f29 * f31 * f43a * f43b * f57, 100, rng, msg='100')
        self.assertFindsFactor(
            f29 * f31 * f43a * f43b * f57, 10_000, rng, msg='10k')
        self.assertFindsFactor(
            f29 * f31 * f43a * f43b * f57, 1_000_000, rng, msg='1M')

    def test_high_smoothness(self) -> None:
        # 32-bit primes
        p = 1 + 2**3 * 3 * 1693 * (ps := 91639)  # 3723475849
        q = 1 + 2 * 3673 * (qs := 450503)  # 3309395039
        r = 1 + 2 * 3 * 13 * (rs := 52199869)  # 4071589783
        # t = 1 + 2 * 3 * 5 * 17 * (ts := 8125987)  # 4144253371
        # u = 1 + 2**9 * 7 * 101 * (us := 11239)  # 4068338177
        self.checkPrimes(p, q, r)

        rng = self.rng(1234567**3)

        self.assertFindsFactor(p * q, min(ps, qs), rng, msg='p*q')
        self.assertFindsFactor(q * r, min(qs, rs), rng, msg='r*q')

        # This is too un-smooth to test quickly
        # # Very un-smooth numbers
        # self.assertFindsFactor(r * t, min(rs, ts), rng, msg='r*t (unsmooth)')

    def test_failures(self) -> None:
        # Test failing cases by feeding b values
        p = 1 + 2 * 179 * 211
        q = 1 + 2 * 173 * 181
        self.checkPrimes(p, q)

        with self.assertRaisesRegex(
                ValueError, r'no factors 1 greater than a \d+-smooth number'):
            ff.find_factor_pm1(p * q, 100, count(2))

        f43a = 1 + 2 * 17 * 19 * 37 * 43    # 20-bit
        f43b = 1 + 2 * 17 * 19 *      43    # 15-bit  # noqa: E222
        self.checkPrimes(f43a, f43b)

        # Numbers with the same smoothness don't perform well
        with self.assertRaisesRegex(ValueError, r'failed to find a factor'):
            ff.find_factor_pm1(n := f43a * f43b, 43, [2, 3, n // 37])

    def rng(self, seed: int):
        """Produce an unseeded Blum-Blum-Shub PRNG.
        The seed should be fairly large.
        """
        # 64-bit primes
        p = 17795126297973966559
        q = 9310741375428223823
        return pr.BlumBlumShub(p, q, seed=seed)


class TestFactors(unittest.TestCase):
    def factors(self, n: int) -> Iterable[int]:
        return ff.factors(n)

    def assertFactors(self, factors, msg=None):
        self.assertEqual(sorted(self.factors(prod(factors))),
                         sorted(factors),
                         msg)

    def test_factors(self):
        self.assertFactors([2, 3, 5])
        self.assertFactors([2, 2, 2, 3, 5, 17, 29, 37, 131, 233])
        self.assertFactors(PRIMES_BELOW_4000[-10:])

    def test_primes(self):
        self.assertFactors([2])
        self.assertFactors([3])
        self.assertFactors([5])
        self.assertFactors([PRIMES_BELOW_4000[-1]])

    def test_special_cases(self):
        self.assertEmpty(self.factors(1))
        self.assertEmpty(self.factors(0))

    def assertEmpty(self, iterator, msg=None):
        """Assert that an iterator or generator is empty.
        """
        try:
            item = next(iterator)
        except StopIteration:
            pass
        else:
            # Note: _formatMessage is an implementation detail
            msg = self._formatMessage(msg, 'Iterator not empty.'
                                      f' First item: {item}')
            raise self.failureException(msg)


class TestUniqueFactors(TestFactors):
    def factors(self, n: int) -> Iterable[int]:
        return ff.unique_factors(n)

    def assertFactors(self, factors, msg=None):
        self.assertEqual(sorted(ff.unique_factors(prod(factors))),
                         sorted(set(factors)),
                         msg)

del FactorizeTestCase
