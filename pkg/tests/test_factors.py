"""
A few simple test cases for my factorization module.

Not comprehensive.
"""
from collections import Counter
from math import prod
import unittest

from homework.factors import (
    Factors,
    factorize,
    totient,
    _generate_unique_factors,
)


class TestOther(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import homework.util
        homework.util.VERBOSE = False

    def test_unique_factors(self):
        self.assertListEqual(list(_generate_unique_factors(29)), [29])

        factors = [2]*3 + [5] + [17]*2
        self.assertListEqual(
            list(_generate_unique_factors(prod(factors))),
            [2, 5, 17])

    def test_totient(self):
        self.assertEqual(totient(2), 1)
        self.assertEqual(totient(7), 6)
        self.assertEqual(totient(29), 28)
        # Composites
        self.assertEqual(totient(4), 2)
        self.assertEqual(totient(35), 24)
        self.assertEqual(totient(60), 16)


class TestFactors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import homework.util
        homework.util.VERBOSE = False

    def assertProd(self, factors: Factors, expect: int):
        self.assertEqual(factors.prod, expect)

    def test_simple(self):
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

    def test_str_repr(self):
        factors = factorize(8)
        self.assertEqual(str(factors), 'Factors(8)')
        self.assertEqual(repr(factors), 'Factors(8, {2: 3})')
        factors = factorize(2**3 * 5**2)
        self.assertEqual(str(factors), 'Factors(200)')
        self.assertIn(repr(factors), ['Factors(200, {2: 3, 5: 2})',
                                      'Factors(200, {5: 2, 2: 3})'])

    def test_errs(self):
        # It's not actually stated in the docs that assertRaises
        # is reusable.
        test = self.assertRaisesRegex(ValueError, r'not prime')
        with test:
            factorize(0)
        with test:
            Factors.of(0)
        with test:
            Factors([0])
        with test:
            Factors([2, 3, 5, 6])

    def test_setitem(self):
        f = factorize(2 * 3 * 5)
        f[3] = 3 # modify factor
        f[7] = 1 # new factor
        f[2] = 0 # delete factor
        self.assertEqual(f.prod, 3**3 * 5 * 7)
        self.assertNotIn(2, f)  # differs from Counter
        with self.assertRaisesRegex(ValueError, r'non-prime factor'):
            f[4] = 1

    def test_gcd(self):
        gcd = 4231
        f1 = factorize(gcd * 2**5 * 17)
        f2 = factorize(n2 := gcd * 3**2 * 13)
        self.assertEqual(f1.gcd(f2), gcd)
        self.assertEqual(f1.gcd(n2), gcd)

    def test_and(self):
        gcd = 4231
        f1 = factorize(n1 := gcd * 2**5 * 17)
        f2 = factorize(n2 := gcd * 3**2 * 13)
        self.assertProd(f1 & f2, gcd)
        self.assertProd(f1 & n2, gcd)
        self.assertProd(n1 & f2, gcd)

    def test_mul_div(self):
        # This could be way more comprehensive.
        f1 = factorize(n1 := 2 * 3 * 5 * 7)
        f2 = factorize(n2 := 3**2 * 5**2 * 13)
        f3 = factorize(n1 * (q13 := 3**2 * 5 * 11))
        # n3 = n1 * q13

        self.assertProd(f1 * n2, n1 * n2)
        self.assertProd(n1 * f2, n1 * n2)
        self.assertProd(f1 * f2, n1 * n2)
        with self.assertRaises(TypeError):
            'X' * f1

        self.assertProd(f1 / f1, 1)
        self.assertProd(f3 / f1, q13)
        self.assertProd(f3 / n1, q13)
        with self.assertRaisesRegex(ValueError, r'not divisible'):
            f3 / f2
        with self.assertRaises(TypeError):
            f3 / 1.0

    def test_imul_idiv(self):
        f1 = factorize(n1 := 2 * 3 * 5 * 7)
        f2 = factorize(n2 := 3**2 * 5**2 * 13)

        f1 *= n2
        self.assertProd(f1, n1 * n2)
        f1 /= n2
        self.assertProd(f1, n1)
        f1 *= f2
        self.assertProd(f1, n1 * n2)
        f1 /= f2
        self.assertProd(f1, n1)

        with self.assertRaisesRegex(ValueError, r'not divisible'):
            f1 /= f2
        self.assertProd(f1, n1)

        with self.assertRaises(TypeError):
            f1 *= 2.0
        self.assertProd(f1, n1)

        with self.assertRaises(TypeError):
            f1 /= 2.0
        self.assertProd(f1, n1)

    def test_add_sub(self):
        f1 = factorize(n1 := 2 * 3 * 5)
        f2 = factorize(n2 := 3**2 * 5**2 * 13)
        f2a = factorize(n2a := 3 * 5)  # factor of f2
        c2 = Counter(f2)

        add = f1 + f2
        self.assertProd(add, n1 * n2)
        self.assertIsInstance(add, Factors)

        sub = f2 - f2a
        self.assertProd(sub, n2 // n2a)
        self.assertIsInstance(sub, Factors)

        # This behavior is why you shouldn't use - as /.
        # It subtracts factors, to a minimum of zero.
        sub = f2 - f1
        self.assertEqual(sub, c2 - Counter(f1))
        self.assertIsInstance(sub, Factors)

        self.assertNotIsInstance(f1 + c2, Factors)
        self.assertNotIsInstance(f1 - c2, Factors)

