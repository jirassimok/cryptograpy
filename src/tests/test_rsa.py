# mypy: check-untyped-defs
import unittest

import homework.rsa as rsa


class TestRSA(unittest.TestCase):
    def end_to_end(self, p, q, e, message):
        """Run an end-to-end test on the encryption.

        Also tests cracking the encryption.
        """
        priv, pub = rsa.keygen(p, q, e)

        ciphertext = rsa.encrypt(pub, message)
        self.assertNotEqual(message, ciphertext, 'encoded')

        decoded = rsa.decrypt(priv, ciphertext)
        self.assertEqual(message, decoded, 'roundtrip')

        cracked = rsa.crack(pub, ciphertext)
        self.assertEqual(message, cracked, 'cracked')

    def test_a_basic(self):
        p = 632823293
        q = 1004229053
        e = 257
        self.end_to_end(p, q, e, 123456)
        self.end_to_end(p, q, e, 7756221)
        self.end_to_end(p, q, e, 5458782)

    def test_a_special(self):
        p = 632823293
        q = 1004229053
        e = 257
        self.end_to_end(p, q, e, p)
        self.end_to_end(p, q, e, q)
        self.end_to_end(p, q, e, e)

    def test_b_basic(self):
        p = 3017610907
        q = 3985905491
        e = 65537
        self.end_to_end(p, q, e, 123456)
        self.end_to_end(p, q, e, 7756221)
        self.end_to_end(p, q, e, 5458782)

    def test_b_special(self):
        p = 3017610907
        q = 3985905491
        e = 65537
        self.end_to_end(p, q, e, p)
        self.end_to_end(p, q, e, q)
        self.end_to_end(p, q, e, e)

