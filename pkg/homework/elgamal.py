"""ElGamal encryption

Key Functions
-------------
crack
ElGamal.encrypt
ElGamal.decrypt

Key Classes
-----------
Key
ElGamal
"""
from typing import NamedTuple

from .euclid import modular_inverse
from .fastexp import pow as fastexp
from .homework4 import discrete_log


class Key(NamedTuple):
    """The public information for one side of an ElGamal message exchange."""
    prime: int
    base: int
    base_to_secret_power: int


class ElGamal:
    """Represents one user in an ElGamal-encrypted exchange.

    See the publish_key, encrypt, and decrypt methods.

    Attributes
    ----------
    prime : int
        The modulus for the group operations.
    base : int
        The base for the exponents.
    base_to_secret : int
        The base raised to this user's secret power in the group.

    Parameters
    ----------
    prime : int
        The public prime modulus.

    Keyword Parameters
    ------------------
    base : int
        The public base for the exponents.
    secret : int
        This user's secret exponent.
    """
    def __init__(self, prime, *, base, secret):
        """Base and secret are currently required because I don't have randoms.
        """
        self.prime = prime
        self.base = base # or primitive_root(prime, smallest=False)
        self._secret = secret
        self.base_to_secret = fastexp(self.base, self._secret, self.prime)

    def publish_key(self):
        """Publish all the public information needed to be sent messages.

        Returns
        -------
        prime : int
            The modulus for the group operations.
        base : int
            The base for the exponents.
        base_to_secret : int
            The base raised to this user's secret power in the group.
        """
        return Key(self.prime, self.base, self.base_to_secret)

    def encrypt(self, recipient_base_to_secret, message):
        """Encrypt a message to send to another user.

        Parameters
        ----------
        recipient_base_to_secret : int
            The recipient's public key (the shared base raised to the
            recipient's secret power in the shared group).
        message : int
            The message to encrypt.
        """
        b_to_both = fastexp(recipient_base_to_secret,
                            self._secret, self.prime)
        return message * b_to_both % self.prime

    def decrypt(self, sender_base_to_secret, ciphertext):
        """Decrypt a message from another user.

        Parameters
        ----------
        sender_base_to_secret : int
            The sender's public key (the shared base raised to the sender's
            secret power in the shared group).
        ciphertext : int
            The message to decrypt.
        """
        p = self.prime
        b_to_s_inv = modular_inverse(sender_base_to_secret, p)
        b_to_both_inv = fastexp(b_to_s_inv, self._secret, p)
        return b_to_both_inv * ciphertext % p


def crack(prime, base,
          sender_base_to_secret,
          recipient_base_to_secret,
          ciphertext):
    """Break ElGamal encryption.

    Parameters
    ----------
    prime : int
        The modulus for the group operations.
    base : int
        The base for the exponents.
    sender_base_to_secret : int
        The sender's public key (the base raised to the sender's secret power).
    recipient_base_to_secret : int
        The recipient's public key (the base raised to the recipient's secret
        power).
    ciphertext : int
        The message to decrypt.
    """
    p, b, c = prime, base, ciphertext
    bs = sender_base_to_secret
    br = recipient_base_to_secret
    r = discrete_log(br, b, p)
    bs_inv = fastexp(bs, p - 2, p)  # b^t^-1
    bsr_inv = fastexp(bs_inv, r, p)  # b^t^r^-1
    return c * bsr_inv % p
