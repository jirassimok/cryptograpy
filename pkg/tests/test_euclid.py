# mypy: check-untyped-defs
from contextlib import redirect_stdout
import contextvars
from math import gcd as math_gcd
import os
import unittest
from unittest.mock import patch

from homework.euclid import (
    # These are the functions tested in this file.
    divmod as manual_divmod,
    euclid,
    euclid2,
    ext_euclid,
    ext_euclid_magic_index,
    ext_euclid_full_columns,
    ext_euclid_full_table,
    PseudoTable,
)


@patch('homework.util.VERBOSE', False)
class TestEuclid(unittest.TestCase):
    """Test the Euclidean algorithm.

    Override gcd_function to test other implementations.
    """
    _params: tuple[
        tuple[int, int, int] # a, b, expected
        | tuple[int, int, int, str] # a, b, expected, message
        | tuple[int, int, str]      # a, b, message (compare to math.gcd)
        , ...
    ] = (
        (1, 1, 1),
        (1, 2, 1),
        (127, 127, 127, 'same prime'), # self prime
        (128, 128, 128, 'same composite'),
        (13, 17, 1, 'small primes'),
        (2, 50, 2, 'small multiple'),
        (17, 17*51312, 17, 'large multiple'),
        (127*2*2*3*13, 127*5*17*23, 127, 'not coprime'),
        (6553, 3651, 1, 'larger primes'),
        (2*(3**8)*17*(227**3), (5**5)*7*7*743*1021, 1, 'coprime'),
        (1243, 5432, 1, 'larger coprimes'),
        (102313, 103927, 'homework example'),
    )

    @staticmethod
    def gcd_function(a: int, b: int, /) -> int:
        """Override this to change the function being tested."""
        return euclid(a, b)

    def gcd(self, a: int, b: int, /) -> int:
        return self.gcd_function(a, b)

    @property
    def params(self):
        """Override this to add params."""
        return self._params

    def setUp(self):
        """Disable verbose mode before testing.
        """
        import homework.util
        homework.util.VERBOSE = False
        import homework.euclid
        homework.euclid.MANUAL_DIVMOD.set(True)

    def test_gcd(self):
        for a, b, expected, *msg in self.filter_params():
            with self.subTest(*msg, a=a, b=b):
                self.assertEqual(self.gcd(a, b), expected)
                # If that passes, also try it with args flipped.
                with self.subTest(*msg, a=b, b=a):
                    self.assertEqual(self.gcd(b, a), expected)

    def test_zero(self):
        for a, b in [(0, 1), (1, 0),
                     (0, 127), (127, 0),
                     (0, 24), (24, 0)]:
            with self.subTest('Zero arg', a=a, b=b):
                self.assertEqual(self.gcd(a, b), a or b)

    # This should be the default, but set it just in case.
    @patch('homework.euclid.ALLOW_NEGATIVE', False)
    def test_negative_errs(self):
        for a, b in [(-24, -45), (-8, -2),
                     (-24, 45), (-8, 2),
                     (24, -45), (8, -2)]:
            with self.subTest('Negative arg', a=a, b=b):
                with self.assertRaises(ValueError):
                    self.gcd(a, b)

    def test_verbose(self):
        """A quick test to get some coverage on the printing code
        and to make sure verbosity doesn't affect correctness.
        """
        # Local setup/teardown for some quick tests
        import homework.util
        homework.util.VERBOSE = True
        try:
            with (open(os.devnull, 'w') as out,
                  redirect_stdout(out)):
                a, b, e = 127*2*2*3*13, 127*5*17*23, 127
                self.assertEqual(self.gcd(a, b), e)
                self.assertEqual(self.gcd(a, 0), a)
                self.assertEqual(self.gcd(0, b), b)
                self.assertEqual(self.gcd(1, 10), 1)
        finally:
            homework.util.VERBOSE = False

    def filter_params(self):
        """Return the basic test parameters with defaults filled in.
        """
        for a, b, *rest in self.params:
            match rest:
                case [int() as g]:
                    yield a, b, g
                case [str() as msg]:
                    yield a, b, math_gcd(a, b), msg
                case [int() as g, str() as msg]:
                    yield a, b, g, msg
                case _:  # pragma: no cover (illegal state)
                    raise TypeError('invalid params')


class TestEuclid2(TestEuclid):
    gcd_function = staticmethod(euclid2)


class TestExtEuclid(TestEuclid2):
    @staticmethod
    def ext_euclid_function(a: int, b: int, /) -> tuple[int, int, int]:
        return ext_euclid(a, b)

    @classmethod
    def gcd(cls, a: int, b: int, /) -> int:
        return cls.ext_euclid_function(a, b)[0]

    def test_one_zero(self):
        for a, b in [(7, 0), (0, 7)]:
            with self.subTest('zero', a=a, b=b):
                gcd, s, t = self.ext_euclid_function(a, b)
                self.assertEqual(s*a + t*b, gcd)
                for coef, arg in ((s, a), (t, b)):
                    if arg == 0:
                        self.assertEqual(coef, 0, msg='0 coefficient = 0')
                    else:
                        self.assertEqual(coef, 1,
                                         msg='nonzero coefficient = 1')

    def test_two_zeros(self):
        a, b = 0, 0
        gcd, s, t = self.ext_euclid_function(a, b)
        # Allow either two zeros as the coefficients
        #  0*0 + 0*0 = 0
        # or the initial values of s and t
        #  1*0 + 0*0 = 0
        self.assertIn((s, t), [(0, 0), (1, 0)])

    def test_coefficients(self):
        """Check that the coefficients from the algorithm are correct.
        """
        for a, b, expected, *msg in self.filter_params():
            with self.subTest(*msg, a=a, b=b):
                gcd, s, t = self.ext_euclid_function(a, b)
                self.assertEqual(s*a + b*t, gcd)


# Something might be wrong here: these don't seem to actually call the
# functions they are supposed to (but TestEuclid2 does, somehow).

class TestExtEuclid2(TestExtEuclid):
    ext_euclid_function = staticmethod(ext_euclid_magic_index)


class TestExtEuclid3(TestExtEuclid):
    ext_euclid_function = staticmethod(ext_euclid_full_table)


class TestExtEuclid4(TestExtEuclid):
    ext_euclid_function = staticmethod(ext_euclid_full_columns)


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


class TestPseudoTable(unittest.TestCase):
    """Quick tests for the PseudoTable class and its friends.

    These aren't near comprehensive, which is fine as long as they aren't used
    in any primary implementations.
    """
    def test_basic(self):
        table = PseudoTable([t1 := 35, t2 := 46], maxlen=2)
        i = table.i
        self.assertEqual(table[i-2], t1)
        self.assertEqual(table[i-1], t2)
        self.assertRaises(IndexError, lambda: table[i-3])

        self.assertRaises(ValueError, lambda: table[0])
        self.assertRaises(ValueError, lambda: table[-1])
        self.assertRaises(ValueError, lambda: table[i])

        table[i] = t3 = 57
        self.assertRaises(ValueError, lambda: table[i])
        self.assertEqual(table[i-1], t3)
        self.assertEqual(table[i-2], t2)
        self.assertRaises(IndexError, lambda: table[i-3])

        with self.assertRaises(ValueError):
            table[0] = 1
        with self.assertRaises(ValueError):
            table[i-1] = 1
        with self.assertRaises(ValueError):
            table[i-2] = 1

    def test_index(self):
        i = PseudoTable.i
        self.assertEqual((i-1).offset, -1)
        self.assertRaises(ValueError, lambda: i+0)
        self.assertRaises(ValueError, lambda: i+1)

    def test_repr(self):
        table = PseudoTable([7, 3], maxlen=2)
        self.assertEqual(repr(table), 'PseudoTable([7, 3], maxlen=2)')
        self.assertEqual(repr(table.i - 1), 'PseudoIndex-1')
