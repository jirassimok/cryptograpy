from collections.abc import Iterator
from contextlib import redirect_stdout
import os
from typing import overload
import unittest

import sympy.ntheory as sn

from homework.homework4 import primitive_root, is_primitive_root, bsgs_log
import homework.util

from .test_fastexp import small_cases, filter_params


class TestPrimitiveRoot(unittest.TestCase):
    @staticmethod
    def primitive_root(p: int, mod: int, /) -> int:
        # Always look for only the smallest primitive root.
        return primitive_root(p, smallest=True)

    _simple_params: tuple[int | tuple[int, int], ...] = (
        (2, 1),
        (4, 3), # I special-cased this in
        (3, 2),
        (5, 2),
        (7, 3),
        37,
        263,
        997,
        2111,
    )
    """Basic test cases.

    Each case is either an (int, int) tuple of argument and expected result,
    or just an argument (an int), which will be compared against
    sympy.primitive_root.
    """

    def setUp(self): # could use setUpClass
        """Disable verbose mode before testing.
        """
        homework.util.VERBOSE = False

    def assertIsRoot(self, p, r):
        self.assertTrue(sn.is_primitive_root(r, p), msg=f'p={p}, root={r}')

    def test_simple(self):
        for arg in self._simple_params:
            if isinstance(arg, int):
                p, expected = arg, sn.primitive_root(arg)
            else:
                p, expected = arg

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
        for p in (457, 1021, 3371, 3863, 10847):
            # Try 10 times to get a primitive root that isn't the smallest.
            r = primitive_root(p, nocheck=True,
                               smallest=False,
                               base_tries=1000)
            if r == sn.primitive_root(p):
                print(f'(accidentally found smallest primitive root of {p})')
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
        p = 10513
        self.assertEqual(primitive_root(p, smallest=False, base_tries=0),
                         sn.primitive_root(p))
        p = 12619
        self.assertEqual(primitive_root(p, smallest=False, base_tries=0),
                         sn.primitive_root(p))


class TestIsPrimitiveRoot(unittest.TestCase):
    _simple_params = TestPrimitiveRoot._simple_params

    def test_simple(self):
        for arg in self._simple_params:
            if isinstance(arg, int):
                p, root = arg, sn.primitive_root(arg)
            else:
                p, root = arg
        self.assertTrue(is_primitive_root(root, p), f'p={p}, root={root}')

    def test_errs(self):
        # non-prime p
        with self.assertRaises(ValueError):
            is_primitive_root(2, 15)

    def test_extended(self):
        for p in (73, 617, 1999):
            for x in range(2, p):
                self.assertEqual(is_primitive_root(x, p),
                                 sn.is_primitive_root(x, p),
                                 f'p={p}, potential root {x}')

    def test_with_factors(self):
        from homework.factors import factorize
        for p in (73, 617, 1999):
            factors = factorize(p - 1)
            for x in range(2, p):
                self.assertEqual(is_primitive_root(x, p, factors=factors),
                                 sn.is_primitive_root(x, p),
                                 f'p={p}, potential root {x}')


class TestBsgsLog(unittest.TestCase):
    # Tests bsgs_log using the small test cases from fastexp.
    # (The large cases are too big to run in a reasonable amount of time.)
    def setUp(self):
        homework.util.VERBOSE = False

    @overload
    def assertLog(self, base, mod, *, power=None): ...

    @overload
    def assertLog(self, base, mod, *, exp=None): ...

    def assertLog(self, base, mod, *, exp=None, power=None):
        # Main test function; tests bsgs_log against builtin pow.
        if power is None:
            power = pow(base, exp, mod)
        self.assertEqual(power, pow(base, bsgs_log(power, base, mod), mod))

    def fastexp_params(self
                       ) -> Iterator[tuple[tuple[int, int, int], int]
                                     | tuple[tuple[int, int, int], int, str]]:
        # Skip non-modular test cases from fastexp
        for args, *rest in filter_params(small_cases()):
            if len(args) < 3 or args[2] is None or not sn.isprime(args[2]):
                # skip cases without a prime modulus
                continue
            yield args, *rest

    def test_basic(self) -> None:
        """Test using the numbers from the fastexp tests."""
        for (base, _, mod), power, *msg in self.fastexp_params():
            with self.subTest(*msg, args=(power, base, mod)):
                self.assertLog(base, mod, power=power)

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
