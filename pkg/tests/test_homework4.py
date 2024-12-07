from collections.abc import Iterator
from contextlib import redirect_stdout
import os
from typing import overload
import unittest

from homework.homework4 import primitive_root, is_primitive_root, bsgs_log
import homework.util

from .test_fastexp import small_cases, filter_params

# If sympy is not available, use alternate tests. These will be slower.
try:
    import sympy.ntheory as _sn
    isprime = _sn.isprime
    compare_is_primitive_root = _sn.is_primitive_root
except ImportError:
    # For isprime, use my own implementation, safe to a little above 2**78
    from homework.pseudoprime import is_prime as isprime

    def compare_is_primitive_root(a, p):
        """Test for primitive roots using repeated multiplication.

        If 1 is encountered before the (p-1)th power, it's not a primitive
        root.

        If p is not prime, this test will not be accurate.
        """
        if a == 1:
            return False
        acc = a
        for exp in range(2, p):
            # for each power or a from a**2 to a**(p-1)
            acc = acc * a % p
            if acc == 1:
                return exp == p - 1  # true only on last loop
        else:
            # Loop finished without finding a 1
            raise ValueError('simple primitive root test failed')


class TestPrimitiveRoot(unittest.TestCase):
    @staticmethod
    def primitive_root(p: int, mod: int, /) -> int:
        # Always look for only the smallest primitive root.
        return primitive_root(p, smallest=True)

    simple_params: tuple[tuple[int, int], ...] = (
        (2, 1),  # 2 is the only number with primitive root 1
        (4, 3),  # I special-cased 4 in
        (3, 2),
        (5, 2),
        (7, 3),
        (41, 6),  # non-prime primitive root of a prime
        (191, 19),
        (263, 5),
        (409, 21),  # another non-prime root
        (997, 7),
        (2111, 7),
    )
    """Basic test cases; pairs of (number, smallest primitive root)."""

    def setUp(self): # could use setUpClass
        """Disable verbose mode before testing.
        """
        homework.util.VERBOSE = False

    def assertIsRoot(self, p, r):
        self.assertTrue(compare_is_primitive_root(r, p),
                        msg=f'p={p}, root={r}')

    def test_simple(self):
        for p, expected in self.simple_params:
            with self.subTest(arg=p):
                self.assertEqual(primitive_root(p, smallest=True), expected)

    def test_errs(self):
        with self.assertRaisesRegex(ValueError, 'no primitive roots'):
            primitive_root(3 * 5)

        with self.assertRaisesRegex(NotImplementedError, 'non-prime'):
            primitive_root(2 * 3**3)
            primitive_root(2**4)

    def test_random_gen(self):
        # Test generating non-smallest primitive roots
        for p, smallest_root in ((457, 13),
                                 (1021, 10),
                                 (3371, 2),
                                 (3863, 5),
                                 (10847, 5)):
            # Try 10 times to get a primitive root that isn't the smallest.
            r = primitive_root(p, nocheck=True,
                               smallest=False,
                               base_tries=1000)
            if r == smallest_root:
                self.fail(f'accidentally found smallest primitive root of {p};'
                          'try again')
            self.assertIsRoot(p, r)

    def test_try_first(self):
        # try_first
        self.assertEqual(primitive_root(2609, try_first=(100, 357, 280, 100)),
                         280)

    def test_fallback_paths(self):
        # Small arguments get random order
        for p in (9419, 3779): # 9419 has 4416 roots, 3779 has 1888
            while 2 == (root := primitive_root(p, smallest=False,
                                               base_tries=0)):
                pass # loop until we find a non-smallest root
            self.assertIsRoot(p, root)

        # Large arguments don't get random order
        p = 10781
        self.assertEqual(primitive_root(p, smallest=False, base_tries=0), 10,
                         p)
        p = 12911
        self.assertEqual(primitive_root(p, smallest=False, base_tries=0), 23,
                         p)


class TestIsPrimitiveRoot(unittest.TestCase):
    simple_params = TestPrimitiveRoot.simple_params

    def test_simple(self):
        for p, root in self.simple_params:
            self.assertTrue(is_primitive_root(root, p), f'p={p}, root={root}')

    def test_errs(self):
        # non-prime p
        with self.assertRaises(ValueError):
            is_primitive_root(2, 15)

    def test_extended(self):
        for p in (73, 617, 1999):
            for x in range(2, p):
                self.assertEqual(is_primitive_root(x, p),
                                 compare_is_primitive_root(x, p),
                                 f'p={p}, potential root {x}')

    def test_with_factors(self):
        from homework.factors import factorize
        for p in (73, 617, 1999):
            factors = factorize(p - 1)
            for x in range(2, p):
                self.assertEqual(is_primitive_root(x, p, factors=factors),
                                 compare_is_primitive_root(x, p),
                                 f'p={p}, potential root {x}')


class TestBsgsLog(unittest.TestCase):
    # Tests bsgs_log using the small test cases from fastexp.
    # (The large cases are too big to run in a reasonable amount of time.)
    def setUp(self):
        homework.util.VERBOSE = False

    @overload
    def assertLog(self, base: int, mod: int, *, power: int): ...

    @overload
    def assertLog(self, base: int, mod: int, *, exp: int): ...

    def assertLog(self, base, mod, *, exp=None, power=None):
        """Assert that bsgs_log(power, base, mod) = exp.

        Actually, this asserts only that bsgs_log finds a correct exponent,
        not that it finds the given exponent in particular, if one is given.

        Exactly one of power and exp should be given.
        (If power is given, exp is not used. If exp is given, it is used
        to find the power.)
        """
        if power is None:
            power = pow(base, exp, mod)
        self.assertEqual(power, pow(base, bsgs_log(power, base, mod), mod))

    def fastexp_params(self
                       ) -> Iterator[tuple[tuple[int, int, int], int]
                                     | tuple[tuple[int, int, int], int, str]]:
        for args, *rest in filter_params(small_cases()):
            if len(args) < 3 or args[2] is None or not isprime(args[2]):
                # skip cases without a prime modulus
                # (and without any modulus)
                continue
            yield args, *rest

    def test_fastexp_params(self) -> None:
        """Test using the numbers from the fastexp tests."""
        for (base, _, mod), power, *msg in self.fastexp_params():
            # The exponent used in fastexp tests isn't necessarily the same one
            # we'll get from bsgs_log, so we have to test it differently.
            with self.subTest(*msg, args=(power, base, mod)):
                self.assertLog(base, mod, power=power)

    def test_basic(self) -> None:
        """A few basic test cases."""
        # The examples from the lecture notes
        args = (3, 2, 101)
        self.assertEqual(bsgs_log(*args), 69, args)
        args = (3, 2, 29)
        self.assertEqual(bsgs_log(*args), 5, args)

    def test_no_log(self):
        with self.assertRaises(ValueError):
            bsgs_log(0, 5, 13)

        with self.assertRaises(ValueError):
            # The modulus isn't prime, but the number given also isn't coprime
            # to it, so we get a specific error. Note that for some inputs,
            # a non-prime modulus will produce non-error results that are not
            # the discrete logarithm.
            bsgs_log(37*3, 2, 43*37)

    def test_verbose(self):
        """Simple test for coverage and correctness in verbose mode."""
        # Local setup/teardown for some quick test
        homework.util.VERBOSE = True
        try:
            with (open(os.devnull, 'w') as out,
                  redirect_stdout(out)):
                self.assertLog(47, 97, exp=8) # 47**8 % 97 == 1
                self.assertLog(43, 127, exp=15)
        finally:
            homework.util.VERBOSE = False
