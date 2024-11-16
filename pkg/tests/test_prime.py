# noqa: E225
from collections.abc import Iterable
from itertools import count, dropwhile, islice, takewhile, zip_longest
import unittest

import homework.prime as prime
import homework.util


def take(iterator, n):
    """Get n items from an iterator as a list.
    """
    return list(islice(iterator, n))


def takebetween(iterable, start, stop):
    """Get values in the range [start, stop) from an iterator.

    Gets the first run of values in the range.
    """
    return takewhile(lambda x: x < stop,
                     dropwhile(lambda x: x < start, iterable))


class TestPrime(unittest.TestCase):
    """Some basic tests for my prime-number utils.

    Doesn't test the fancier features like making sure the cache.
    """
    def setUp(self):
        self.sympy_token = homework.util.USE_SYMPY.set(False)

    def tearDown(self):
        homework.util.USE_SYMPY.reset(self.sympy_token)
        self.sympy_token = None

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

    def assertIterEqual(self, gen1: Iterable, gen2: Iterable, msg=None):
        canary = object()
        for i, a, b in zip_longest(count(), gen1, gen2, fillvalue=canary):
            if a is canary and b is canary:
                break  # both ended; test passed
            elif a is canary:
                raise self.failureException(
                    f'left iterator ended before right: {msg}')
            elif b is canary:
                raise self.failureException(
                    f'right iterator ended before left: {msg}')
            elif a != b:
                raise self.failureException(
                    f'iterator mismatch at position {i}: {a} != {b} [{msg}]')

    def test_primes(self):
        self.assertEqual(list(takewhile(lambda p: p < 4000, prime.primes())),
                         PRIMES_BELOW_4000)

    def test_is_prime(self):
        is_prime = prime.is_prime

        for n in range(-5, PRIMES_BELOW_4000[-1]+1):
            self.assertEqual(is_prime(n),
                             n in PRIMES_BELOW_4000,
                             f'is_prime({n})')

    def test_primerange(self):
        primerange = prime.primerange

        def pr(start=None, stop=None):
            return list(primerange(start, stop))

        # Simple cases
        self.assertEqual(pr(10), [2, 3, 5, 7], 'just end')
        self.assertEqual(pr(-5, 10), [2, 3, 5, 7], 'negative start')
        self.assertEqual(pr(5, 17), [5, 7, 11, 13], 'both bounds prime')
        self.assertEqual(pr(6, 18), [7, 11, 13, 17], 'neither bound prime')
        self.assertEqual(pr(23, 24), [23], 'size-one prime range')

        # Empty ranges
        self.assertEmpty(primerange(2), 'empty [x, 2)')
        self.assertEmpty(primerange(-10, 2), 'empty [negative, 2)')
        self.assertEmpty(primerange(0, 2), 'empty [0, 2)')
        self.assertEmpty(primerange(8, 11), 'empty [8, 11)')
        # start/neither/end are prime
        self.assertEqual(pr(89, 96), [89], 'start at only prime')
        self.assertEmpty(primerange(90, 96), 'empty [90, 96)')
        self.assertEmpty(primerange(90, 97), 'empty [90, 97)')

        # Bad ranges
        self.assertEmpty(primerange(0), 'bad range [x, 0)')
        self.assertEmpty(primerange(-5), 'bad range [x, -5)')
        self.assertEmpty(primerange(-10, -1), 'bad range [-10, 2)')
        self.assertEmpty(primerange(10, 1), 'bad range [10, 1)')

        self.assertEqual(pr(4000), PRIMES_BELOW_4000)
        self.assertIterEqual(primerange(2000, 3000),
                             takebetween(PRIMES_BELOW_4000, 2000, 3000),
                             '[2000, 3000)')
        self.assertIterEqual(primerange(2000, 3000),
                             takebetween(PRIMES_BELOW_4000, 2000, 3001),
                             'high range, end before prime')
        self.assertIterEqual(primerange(2003, 3000),
                             takebetween(PRIMES_BELOW_4000, 2003, 3000),
                             'high range, start at prime')

    def test_parallel_cache(self):
        """You can use multiple instances of the generator with the same cache.
        """
        primes = prime.primes
        lowprimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

        cache = prime.PrimeCache(lowprimes[:6]) # same as the default cache
        # What is taken and in what order:
        g1 = primes(_cache=cache)
        self.assertEqual(take(g1, i1:=3), lowprimes[:i1]) # get first 3
        g2 = primes(_cache=cache)
        self.assertEqual(take(g2, i2:=6), lowprimes[:i2]) # first 6 (none new)
        self.assertEqual(take(g1, j1:=5), lowprimes[i1:i1+j1]) # 4 to 8 (2 new)
        g3 = primes(_cache=cache)
        self.assertEqual(take(g1, len(lowprimes)-(i1+j1)), lowprimes[i1+j1:])
        self.assertEqual(take(g2, len(lowprimes)-i2), lowprimes[i2:])
        self.assertEqual(take(g3, len(lowprimes)), lowprimes)

    def test_cache(self):
        # Some tests for the cache's direct behaviors.
        cache = prime.PrimeCache([2, 3])
        with self.assertRaises(ValueError):
            cache[0]
        with self.assertRaises(ValueError):
            cache[1]
        with self.assertRaises(ValueError):
            cache[-2]

        it = iter(cache)
        self.assertEqual(next(it), 2)
        cache.add_largest(5)
        self.assertEqual(next(it), 3)
        self.assertEqual(next(it), 5)
        cache.add_largest(7)
        self.assertEqual(next(it), 7)
        with self.assertRaises(StopIteration):
            next(it)
        with self.assertRaises(StopIteration):
            next(it)


class TestSieve(unittest.TestCase):
    def setUp(self):
        self.sympy_token = homework.util.USE_SYMPY.set(False)

    def tearDown(self):
        homework.util.USE_SYMPY.reset(self.sympy_token)
        self.sympy_token = None

    def test_basic(self):
        sieve = prime.Sieve(4000)
        for a, e in zip(sieve.generate(), PRIMES_BELOW_4000):
            self.assertEqual(a, e)

    def test_extended(self):
        sieve = prime.Sieve(100)
        # Before populating the sieve, all odds look prime.
        for i, isprime in enumerate(sieve):
            if i > 2:
                self.assertEqual(isprime, bool(i % 2), f'{i} initial')

        sieve.populate()
        # Test sieve contents
        for i, isprime in enumerate(sieve):
            self.assertEqual(isprime, i in PRIMES_BELOW_4000, f'{i} present')

        # Generate still works after populating
        for a, e in zip(sieve.generate(), PRIMES_BELOW_4000):
            self.assertEqual(a, e, f'{i} re-generate')

    def test_getitem(self):
        sieve = prime.Sieve(100)
        sieve.populate()
        self.assertEqual(sieve[2], True)
        self.assertEqual(sieve[10], False)
        self.assertEqual(sieve[29], True)
        with self.assertRaisesRegex(TypeError, 'must be int'):
            sieve[1.0]
        with self.assertRaises(IndexError):
            sieve[100]

    def test_setitem(self):
        sieve = prime.Sieve(100)
        self.assertEqual(sieve[9], True)
        sieve[9] = False
        self.assertEqual(sieve[9], False)
        sieve[9] = True  # This shouldn't be allowed.
        self.assertEqual(sieve[9], True)

        with self.assertRaises(TypeError):
            sieve[9] = 0
        with self.assertRaises(ValueError):
            sieve[8] = False

    def test_extra(self):
        # Tests for some internal and/or unused functions/code paths.
        sieve = prime.Sieve(100)
        with self.assertRaises(ValueError):
            sieve._find_true(0)
        with self.assertRaises(ValueError):
            sieve._find_true(1)
        with self.assertRaises(ValueError):
            sieve._find_true(2)

        self.assertEqual(prime.Sieve._index_of(5), 2)
        self.assertEqual(prime.Sieve._index_of(99), 49)
        with self.assertRaises(ValueError):
            prime.Sieve._index_of(4)


class TestBits(unittest.TestCase):
    def test_bit_masks(self):
        """Extra tests for coverage of methods not used in the Sieve.

        These tests are insufficient to prove correctness.
        """
        bits = prime.bits
        self.assertEqual(bits.low_bits(0b11010111, 5), 0b00010111)
        self.assertEqual(bits.high_bits(0b11010111, 5), 0b11010000)
        self.assertEqual(bits.clear_low_bits(0b11100110, 6), 0b11000000)
        self.assertEqual(bits.clear_high_bits(0b11100110, 6), 0b00000010)

        self.assertEqual(bits.zero_at2(0) & 0xFF, 0b11111110, 0)
        self.assertEqual(bits.zero_at2(7) & 0xFF, 0b01111111, 7)
        self.assertEqual(bits.zero_at2(3) & 0xFF, 0b11110111, 3)

PRIMES_BELOW_4000 = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
    331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
    613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
    709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
    821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911,
    919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
    1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
    1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181,
    1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277,
    1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361,
    1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451,
    1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531,
    1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609,
    1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699,
    1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789,
    1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889,
    1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997,
    1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083,
    2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161,
    2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273,
    2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357,
    2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441,
    2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551,
    2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663,
    2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729,
    2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819,
    2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917,
    2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023,
    3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137,
    3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251,
    3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331,
    3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449,
    3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533,
    3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617,
    3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709,
    3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821,
    3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917,
    3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989,
]
