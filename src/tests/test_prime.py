# noqa: E225
from itertools import takewhile

from .util import PRIMES_BELOW_4000, take, takebetween, TestCase

import homework.prime as prime


class TestPrime(TestCase):
    """Some basic tests for my prime-number utils.
    """
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


class TestPrimeCache(TestCase):
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

        self.assertEqual(cache[-1], 3)

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

        self.assertEqual(cache[-1], 7)
