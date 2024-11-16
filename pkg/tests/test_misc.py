"""
Tests for miscellaneous support code.
"""
from collections import Counter
import contextvars
from itertools import takewhile
from math import prod
import unittest

from homework.euclid import divmod as manual_divmod
from homework.factors import factorize
from homework.prime import primes

from .test_prime import PRIMES_BELOW_4000


class TestManualDivmod(unittest.TestCase):
    token: contextvars.Token[bool] | None = None

    @classmethod
    def setUpClass(cls):
        import homework.euclid
        cls.token = homework.euclid.MANUAL_DIVMOD.set(True)

    @classmethod
    def tearDownClass(cls):
        import homework.euclid
        homework.euclid.MANUAL_DIVMOD.reset(cls.token)
        cls.token = None

    def test_basic(self):
        self.assertEqual(manual_divmod(123, 1), (123, 0))
        self.assertEqual(manual_divmod(9, 9), (1, 0))
        self.assertEqual(manual_divmod(3, 8), (0, 3))
        self.assertEqual(manual_divmod(30, 5), (6, 0))
        self.assertEqual(manual_divmod(18, 7), (2, 4))
        self.assertEqual(manual_divmod(37*94 + 18, 37), (94, 18))

    def test_errs(self):
        with self.assertRaises(ValueError):
            manual_divmod(-1, 3)
        with self.assertRaises(ValueError):
            manual_divmod(3, -2)
        with self.assertRaises(ZeroDivisionError):
            manual_divmod(3, 0)


class TestSympyAlternates(unittest.TestCase):
    """Test for the alternate versions of functions based on USE_SYMPY setting.
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
