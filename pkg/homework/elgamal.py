from typing import NamedTuple

from .fastexp import fastexp as _fastexp
from .homework4 import bsgs_log as _bsgs_log
from .euclid import ext_euclid as ext_euclid


# Always use non-verbose functions here.

def fastexp(base: int, exp: int, modulus: int | None = None) -> int:
    return _fastexp(base, exp, modulus, verbose=False)


def bsgs_log(x: int, base: int, modulus: int):
    return _bsgs_log(x, base, modulus, verbose=False)


def inverse(n: int, modulus: int):
    return ext_euclid(modulus, n, verbose=False)[-1] % modulus


class Key(NamedTuple):
    prime: int
    base: int
    base_to_secret_power: int


class User:
    def __init__(self, prime, *, base, secret):
        """Base and secret are currently required because I don't have randoms.
        """
        self.prime = prime
        self.base = base # or primitive_root(prime, smallest=False)
        self._secret = secret
        self.base_to_secret = fastexp(self.base, self._secret, self.prime)

    def publish_key(self):
        return Key(self.prime, self.base, self.base_to_secret)

    def encrypt(self, recipient_base_to_secret, message):
        b_to_both = fastexp(recipient_base_to_secret,
                            self._secret, self.prime)
        return message * b_to_both % self.prime

    def decrypt(self, sender_base_to_secret, ciphertext):
        p = self.prime
        b_to_s_inv = inverse(sender_base_to_secret, p)
        b_to_both_inv = fastexp(b_to_s_inv, self._secret, p)
        return b_to_both_inv * ciphertext % p


def crack(prime, base,
          sender_base_to_secret,
          recipient_base_to_secret,
          ciphertext):
    p, b, c = prime, base, ciphertext
    bs = sender_base_to_secret
    br = recipient_base_to_secret
    r = bsgs_log(br, b, p)
    bs_inv = fastexp(bs, p - 2, p)  # b^t^-1
    bsr_inv = fastexp(bs_inv, r, p)  # b^t^r^-1
    return c * bsr_inv % p
