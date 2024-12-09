# mypy: check-untyped-defs
import unittest

import homework.elgamal as elg


class TestElgamal(unittest.TestCase):
    def end_to_end(self, prime, base,
                   sender_secret, recipient_secret, message):
        """Run an end-to-end test on the encryption.

        Also tests cracking the encryption.
        """
        sender = elg.ElGamal(prime, base, sender_secret)
        sender_key = sender.publish_key()

        recipient = elg.ElGamal(sender_key.prime,
                                sender_key.base,
                                recipient_secret)
        recipient_key = recipient.publish_key()

        ciphertext = sender.encrypt(recipient_key.power, message)

        decoded = recipient.decrypt(sender_key[2], ciphertext)

        self.assertNotEqual(message, ciphertext, 'encoded')
        self.assertEqual(message, decoded, 'roundtrip')

        prime, base, sender_power = sender_key
        cracked = elg.crack(prime, base, sender_power,
                            recipient_key.power, ciphertext)
        self.assertEqual(message, cracked, 'cracked')

    def test_basic(self):
        p = 632823293
        root = 345542487
        secrets = 41156121, 12344511
        self.end_to_end(p, root, *secrets, 123456)
        self.end_to_end(p, root, *secrets, 7756221)
        self.end_to_end(p, root, *secrets, 5458782)

    def test_example_alice(self):
        """Test the examples from my report.

        Some of these numbers were generated with these same algorithms,
        but they worked with my group's algorithms without needing any
        corrections, so I still consider them useful.
        """
        # Inputs
        prime = 558755221
        base = 245325847
        sender_secret = 396825982
        message = 123454321
        # Actions
        sender = elg.ElGamal(prime, base, sender_secret)
        pubkey = sender.publish_key()
        self.assertEqual(pubkey.power, 450328945, 'power')
        ciphertext = sender.encrypt(503192593, message)
        self.assertEqual(59923868, ciphertext, 'ciphertext')

    def test_example_bob(self):
        # Inputs
        prime = 601
        base = 2
        sender_power = 526
        recipient_secret = 270
        ciphertext = 551
        # Actions
        recipient = elg.ElGamal(prime, base, recipient_secret)
        self.assertEqual(recipient.power, 432, 'power')
        message = recipient.decrypt(sender_power, ciphertext)
        self.assertEqual(486, message, 'message')

    def test_example_eve(self):
        # Inputs
        prime = 719866891
        base = 573107670
        sender_power = 265302985
        recipient_power = 575640003
        ciphertext = 88756902
        # Actions
        message = elg.crack(prime, base, sender_power,
                            recipient_power, ciphertext)
        self.assertEqual(message, 72105, 'message')
