"""RSA encryption

RSA is implemented here as a collection of methods that operate on tuple-valued
keys (NamedTuples are provided for convenience). In the key tuples, the modulus
(n) is always the first value, and the exponent is the second.

Key Functions
-------------
keygen
encrypt
decrypt
crack
"""
from typing import NamedTuple

from .euclid import gcd, modular_inverse
from .fastexp import pow as fastexp
from .factor import find_factor_rho
from .pseudorandom import PRNG


class RsaKey(NamedTuple):
    modulus: int
    exp: int

    @property
    def n(self):
        return self.modulus


class PublicKey(RsaKey):
    @property
    def e(self):
        return self.exp


class PrivateKey(RsaKey):
    @property
    def d(self):
        return self.exp


def keygen(p: int, q: int, e: int | PRNG) -> tuple[PrivateKey, PublicKey]:
    """Generate keys for use in RSA.

    Parameters
    ----------
    p : int
        The first secret prime.
    q : int
        The second secret prime.
    e : int or pseudorandom.PRNG
        Either an integer to use as the public encryption key, or a PRNG to
        generate one.
    """
    modulus = p * q

    order = (p - 1) * (q - 1)

    if isinstance(e, int):
        if gcd(e, order) != 1:
            raise ValueError('secret key is not coprime to (p-1)*(q-1)')
    else:
        rng = e
        e = rng.randrange(2, order)
        while gcd(e, order) != 1:
            e = rng.randrange(2, order)

    d = modular_inverse(e, order)

    return PrivateKey(modulus, d), PublicKey(modulus, e)


def encrypt(key: PublicKey, message: int) -> int:
    """Encrypt a message using the given public key.

    Parameters
    ----------
    key : PublicKey or pair of ints (modulus, exponent)
        The public key to use for encryption.
    message : int
        The message to encrypt.
    """
    return fastexp(message, key.e, key.n)


def decrypt(key: PrivateKey, ciphertext: int) -> int:
    """Decrypt a ciphertext using the given private key.


    Parameters
    ----------
    key : PrivateKey or pair of ints (modulus, exponent)
        The private key to use for encryption.
    ciphertext : int
        The message to decrypt.
    """
    return fastexp(ciphertext, key.d, key.n)


def crack(key: PublicKey, ciphertext: int) -> int:
    """Decrypt a ciphertext by factoring the public key's modulus.

    Uses Pollard's rho algorithm.

    Parameters
    ----------
    key : PublicKey or pair of ints (modulus, exponent)
        The public key to crack.
    ciphertext : int
        The message to decrypt.
    """
    n, e = key

    p = find_factor_rho(n)
    q = n // p

    # Use the known public key to generate the private key.
    privkey, _ = keygen(p, q, e=e)

    return decrypt(privkey, ciphertext)
