from collections import Counter
from collections.abc import Iterator
from math import prod, isqrt

from .prime import is_prime, primerange


__all__ = ['factorize', 'totient', 'Factors']


def factorize(n, /):
    """Get prime factorization of n.

    Returns a Counter mapping factors to their
    exponent in the factorization.
      n == prod(p**k for p, k in factorize(n).items())

    See the Factors class for details.
    """
    return Factors.of(n)


# There's an itertools recipe for totient that is probably more efficient.
def totient(n, /):
    if is_prime(n):
        return n - 1
    f = factorize(n)
    return prod(p**(k - 1) * (p - 1)
                for p, k in f.items())


def _generate_prime_factors(n) -> Iterator[int]:
    """Generate the prime factors of n.
    """
    for p in primerange(isqrt(n) + 1):
        while n % p == 0:
            yield p
            n //= p
            if n == 1:
                return
    yield n # didn't reach n == 1


def _generate_unique_factors(n) -> Iterator[int]:
    """Generate the unique prime factors of n.
    """
    last = 0
    for f in _generate_prime_factors(n):
        if f != last:
            last = f
            yield f


class Factors(Counter):
    """Counter that represents integer factors.

    Factors with zero occurences are removed from
    the mapping (no entry will ever have value zero).
    Non-prime factors are broken down if added (though
    they can not be set directly).

    Multiplication and (true) division are defined between
    this class and other instances, as well as ints.

    The intersection operator (&) also has been extended
    to allow integer arguments, as the common factor
    operator.

    Addition, subtraction, and intersection (&) of Factors
    works as for Counters, but return Factors if both
    operands are Factors. Intersection also allows integer
    operands as though they were Factors.

    Except as noted above, all Counter operations are
    unchanged and may still return base Counters.

    Some base Counter operations may be highly inefficient
    when returning Factors.
    """
    # Rejected:
    # Make composites return their factor counts, too.

    def __init__(self, iterable=None, _safe=False, **k):
        if not _safe and iterable is not None:
            for n in set(iterable):
                if not is_prime(n):
                    raise ValueError(f'factor {n} is not prime')
        super().__init__(iterable, **k)

    @classmethod
    def of(cls, n, /):
        if n == 0:
            raise ValueError('factor 0 is not prime')
        factors = cls()
        for p in _generate_prime_factors(n):
            super(cls, factors).__setitem__(p, factors[p] + 1)
        return factors

    @classmethod
    def _normalize(cls, n):
        """Normalize int as factors.
        """
        if isinstance(n, int):
            return cls.of(n)
        else:
            return n

    @property
    def prod(self):
        """Get the product of the factors.
        """
        # return prod(self.elements())
        return prod(p**k for p, k in self.items())

    def __str__(self):
        cls = type(self).__name__
        prod = self.prod
        return f'{cls}({prod})'

    def __repr__(self):
        cls = type(self).__name__
        prod = self.prod
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f'{cls}({prod}, {{{items}}})'

    def __setitem__(self, item, value):
        if value == 0:
            del self[item]
        elif item in self or is_prime(item):
            return super().__setitem__(item, value)
        else:
            raise ValueError(f'can not set non-prime factor {item}')

    def gcd(self, other):
        other = self._normalize(other)
        return (self & other).prod

    def __mul__(self, other):
        other = self._normalize(other)
        if isinstance(other, type(self)):
            return self + other
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._normalize(other)
        if isinstance(other, type(self)):
            if other <= self:
                return self - other
            else:
                raise ValueError(f'{self} not divisible by {other}')
        else:
            return NotImplemented

    def __imul__(self, other):
        other = self._normalize(other)
        if isinstance(other, type(self)):
            self += other
            return self
        else:
            return NotImplemented

    def __itruediv__(self, other):
        other = self._normalize(other)
        if isinstance(other, type(self)):
            if other <= self:
                self -= other
                return self
            else:
                raise ValueError(f'{self} not divisible by {other}')
        else:
            return NotImplemented

    @classmethod
    def _wrap_op(cls, other, result):
        if isinstance(other, cls):
            return cls(result, _safe=True)
        else:
            return result

    def __add__(self, other):
        result = super().__add__(other)
        return self._wrap_op(other, result)

    def __sub__(self, other):
        result = super().__sub__(other)
        return self._wrap_op(other, result)

    # Only useful logical operator: common factors
    def __and__(self, other):
        other = self._normalize(other)
        result = super().__and__(other)
        return self._wrap_op(other, result)

    def __rand__(self, other):
        return self.__and__(other)
