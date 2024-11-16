"""
Tests for miscellaneous support code.
"""
from collections import Counter
import contextvars
from itertools import dropwhile, takewhile
from math import prod
import unittest

from homework.factors import factorize
from homework.prime import is_prime, primes, primerange

from .test_prime import PRIMES_BELOW_4000


class TestSympyAlternates(unittest.TestCase):
    """Test for the alternate versions of functions based on USE_SYMPY setting.

    These are just basic checks for sanity and coverage purposes.
    """
    token: contextvars.Token[bool] | None = None

    @classmethod
    def setUpClass(cls):
        import homework.util
        cls.token = homework.util.USE_SYMPY.set(True)

    @classmethod
    def tearDownClass(cls):
        import homework.util
        homework.util.USE_SYMPY.reset(cls.token)
        cls.token = None

    def test_factorize(self):
        # Composite
        true_factors = [2]*3 + [5] + [17]*2
        factors_counter = Counter(true_factors)
        factors_obj = factorize(prod(true_factors))
        self.assertDictEqual(factors_obj, factors_counter)
        self.assertEqual(factors_obj.prod, prod(true_factors))
        # Prime
        factors = factorize(29)
        self.assertDictEqual(factors, {29: 1})
        self.assertEqual(factors.prod, 29)

    def test_primes(self):
        self.assertEqual(list(takewhile(lambda p: p < 4000, primes())),
                         PRIMES_BELOW_4000)

    def test_primerange(self):
        self.assertEqual(list(dropwhile(lambda p: p < 1000,
                                        takewhile(lambda p: p < 4000,
                                                  primerange(1000, 3001)))),
                         [p for p in PRIMES_BELOW_4000 if 1000 <= p < 3001])

    def test_is_prime(self):
        for p in range(4000):
            self.assertEqual(is_prime(p), p in PRIMES_BELOW_4000, p)
