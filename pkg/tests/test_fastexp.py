# mypy: check-untyped-defs
# Don't bother trying to make pyright check this file.
from collections.abc import Iterable, Iterator
from contextlib import redirect_stdout
from math import log10, floor
import os
import sys
from typing import Self
import unittest

import homework.util
from homework.fastexp import (
    # These are the functions tested in this file.
    fastexp,
    fastexp2,
    fastexp_recursive,
    verbose2_fastexp,
    slowexp,
)

type ExpArgs = tuple[int, int] | tuple[int, int, None] | tuple[int, int, int]
"""Tuple of arguments to fastexp.
"""

type CaseArgs = (
    tuple[ExpArgs, int] # args, expected
    | tuple[ExpArgs]    # args (compare to builtin pow)
    | tuple[ExpArgs, int, str] # args, expected, label
    | tuple[ExpArgs, str]      # args, label (compare to builtin pow)
)
"""Args with optional expected value and label.
"""


def ndigits(n):
    if n == 0:
        return 1
    return floor(log10(n)) + 1


## Testing parameters
# These functions provide the test parameters for TestFastExp.

def large_cases() -> Iterable[CaseArgs]:
    """Construct a number of cases with fairly large arguments.

    These test cases do not include expected results; they must be compared to
    an existing implementation.
    """
    from itertools import permutations

    p3 = 127
    p60 = 529247038585542108568540290995084860068177551640322543102099
    p40 = 5365152086379702330152001265920318993223
    p30 = 491677946950317334566508347041

    d = ndigits
    return (
        ((p3, 11*73*410587, p60), 'initial case'),
        ((2, p60, p40), '2 to a big power'),
        *(((a, b, c), f'big primes p{d(a)}**p{d(b)} % p{d(c)}')
          for a, b, c in permutations((p30, p40, p60))),
        # that -> 351420569476919220033676836688505908233185988390868705944354
        ((p3, 3*3*286199), 'non-mod large result'), # result is 5418959 digits
    )


def small_cases() -> Iterator[CaseArgs]:
    """Some basic test cases for modular exponentiation.
    """
    cases: list[tuple[tuple[int, int, int], int, str]
                | tuple[tuple[int, int, int], str]] = [
        ((4, 2, 19), 16, "mod doesn't wrap"),
        ((4, 57, 19), 7, "mod wraps"),
        ((2, 58, 59), 1, 'generator^order'),
        ((7, 10, 7), 0, 'zero'),
        ((2, 2**20, 37), 'super-even power'),
        ((5, 7, 16), 13, 'non-prime modulus'),
        ((5, 8, 16), 1, "non-prime modulus (Euler's theorem)"), # tot(16) = 8
    ]

    for case in cases:
        yield case
        # Make a copy of each case, with a much larger quotient
        (base, exp, mod), *rest = case
        yield (base + base * mod, exp, mod), *rest

    # No-modulus cases
    yield ((5, 4), 625, 'no modulus')
    yield ((3, 16), 43046721, 'no modulus')


## Test classes

class TestFastexp(unittest.TestCase):
    @staticmethod
    def exp(a: int, b: int, mod: int | None = None, /) -> int:
        return fastexp(a, b, mod)

    _params: tuple[CaseArgs, ...] = (*small_cases(), *large_cases())
    """Basic test cases.

    A 3-tuple of ints is the arguments to exp, to compare against builtin pow.
    For anything else, the first entry is a tuple of exp arguments,
    and the other entries may be an expected-value int and a test label (str).
    """

    def setUp(self): # could use setUpClass
        """Disable verbose mode before testing.
        """
        homework.util.VERBOSE = False

    def test_basic(self):
        for args, expected, *msg in self.filter_params():
            with self.subTest(*msg, args=args):
                self.assertEqual(self.exp(*args), expected)

    def test_negative_errs(self):
        with self.assertRaises(ValueError):
            self.exp(1, -1, 2)
        with self.assertRaises(ValueError):
            self.exp(2, -3, 13)

    def test_zero_modulus_errs(self):
        with self.assertRaises(ZeroDivisionError):
            self.exp(2, 1, 0)
        with self.assertRaises(ZeroDivisionError):
            self.exp(1, 2, 0)

    def filter_params(self) -> Iterator[tuple[ExpArgs, int]
                                        | tuple[ExpArgs, int, str]]:
        """Return the basic test parameters with defaults filled in.
        """
        for case_args in self._params:
            # Have to manually unpack by slice to get good typing
            args, rest = case_args[0], case_args[1:]
            match rest:
                case []:
                    yield args, pow(*args)
                case [int() as e]:
                    yield args, e
                case [str() as label]:
                    yield args, pow(*args), label
                case [int() as e, str() as label]:
                    yield args, e, label
                case _:
                    raise TypeError('invalid params')


# I used a separate implementation for the verbose version,
# so it gets its own tests.
class TestFastexpVerbose(TestFastexp):
    def setUp(self):
        # Redirect stdout to the null device so we don't actually have to see
        # all the outputs.
        out = self.enterContext(open(os.devnull, 'w'))
        self.enterContext(redirect_stdout(out))
        homework.util.VERBOSE = True

    def filter_params(self: Self):
        # Skip cases where the result is too big to convert to a string.
        limit = sys.get_int_max_str_digits()
        for args, expected, *label in super().filter_params():
            if expected == 0 or ndigits(expected) <= limit:
                yield args, expected, *label


class TestFastexpVerbose2(TestFastexpVerbose):
    @staticmethod
    def exp(a: int, b: int, mod: int | None = None, /) -> int:
        return verbose2_fastexp(a, b, mod)


class TestFastexp2(TestFastexp):
    @staticmethod
    def exp(a: int, b: int, mod: int | None = None, /) -> int:
        return fastexp2(a, b, mod)


class TestFastexpRecursive(TestFastexp):
    @staticmethod
    def exp(a: int, b: int, mod: int | None = None, /) -> int:
        return fastexp_recursive(a, b, mod)


class TestSlowexp(TestFastexp):
    @staticmethod
    def exp(a: int, b: int, mod: int | None = None, /) -> int:
        return slowexp(a, b, mod)

    def filter_params(self: Self):
        for args, e, *label in super().filter_params():
            # mypy thinks that a in args is int|None.
            if all(a < 10**20 for a in args) and e < 10**20: # type: ignore
                # Skip cases with huge numbers
                yield args, e, *label
