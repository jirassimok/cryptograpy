from .test_prime import PRIMES_BELOW_4000, takebetween, TestCase

from homework.sieve import Sieve, bits


class TestSieve(TestCase):
    def test_basic(self):
        sieve = Sieve(4000)
        self.assertIterEqual(sieve.generate(), PRIMES_BELOW_4000)

    def test_extended(self):
        size = 300
        sieve = Sieve(size)
        # Before populating the sieve, all odds look prime (and evens don't).
        for i, isprime in enumerate(sieve):
            if i > 2:
                self.assertEqual(isprime, bool(i % 2), f'{i} initial')

        # The lowest values look accurate even before being populated.
        self.assertEqual(sieve[0], False)
        self.assertEqual(sieve[1], False)
        self.assertEqual(sieve[2], True)

        sieve.populate()
        # Test sieve contents
        for i, isprime in enumerate(sieve):
            self.assertEqual(isprime, i in PRIMES_BELOW_4000, f'{i} present')

        # Generate still works after populating
        # Generate still works after populating
        self.assertIterEqual(sieve.generate(),
                             takebetween(PRIMES_BELOW_4000, 0, size),
                             're-generate')

    def test_getitem(self):
        sieve = Sieve(100)
        sieve.populate()
        self.assertEqual(sieve[2], True)
        self.assertEqual(sieve[10], False)
        self.assertEqual(sieve[29], True)
        with self.assertRaisesRegex(TypeError, 'must be int'):
            sieve[1.0]
        with self.assertRaises(IndexError):
            sieve[100]

    def test_extra(self):
        # Tests for some internal and/or unused functions/code paths.
        sieve = Sieve(100)
        with self.assertRaises(ValueError):
            sieve._find_true(0)
        with self.assertRaises(ValueError):
            sieve._find_true(1)
        with self.assertRaises(ValueError):
            sieve._find_true(2)

        self.assertEqual(Sieve._index_of(5), 2)
        self.assertEqual(Sieve._index_of(99), 49)
        with self.assertRaises(ValueError):
            Sieve._index_of(4)


class TestBits(TestCase):
    def test_bit_masks(self):
        """Extra tests for coverage of methods not used in the Sieve.

        These tests are insufficient to prove correctness.
        """
        self.assertEqual(bits.low_bits(0b11010111, 5), 0b00010111)
        self.assertEqual(bits.high_bits(0b11010111, 5), 0b11010000)
        self.assertEqual(bits.clear_low_bits(0b11100110, 6), 0b11000000)
        self.assertEqual(bits.clear_high_bits(0b11100110, 6), 0b00000010)

        self.assertEqual(bits.zero_at2(0) & 0xFF, 0b11111110, 0)
        self.assertEqual(bits.zero_at2(7) & 0xFF, 0b01111111, 7)
        self.assertEqual(bits.zero_at2(3) & 0xFF, 0b11110111, 3)
