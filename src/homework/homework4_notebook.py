#!/usr/bin/env python
# coding: utf-8
# flake8: noqa
# type: ignore

# In[3]:

from __future__ import annotations
from collections import Counter
from collections.abc import Generator
from math import floor, prod, sqrt
import random

import sympy
import sympy.ntheory as sn

CHECK_PRIMES = False
"""Whether certain functions should perform primality tests on their arguments.
"""

# Can't use this because you can't do
#   pow(int, sympy.Integer, int)
def sympize(fn):
    """Make a function output its output in sympy.Integer.

    Also works on generator functions.
    """
    from functools import wraps
    import inspect
    if inspect.isgeneratorfunction(fn):
        @wraps(fn)
        def wrapper(*a, **k):
            yield from map(sympy.Integer,
                           fn(*a, **k))
        return wrapper
    else:
        @wraps(fn)
        def wrapper(*a, **k):
            return sympy.Integer(fn(*a, **k))
        return wrapper

def sn_primes(_s = sn.Sieve(), /):
    """Generate primes."""
    from itertools import count
    from sympy import sieve
    for i in count(1):
        yield sieve[i]

def candidate_primes():
    """Generate candidate primes.

    Actually generates small primes,
    then numbers congruent to 5 or 1 mod 6.
    """
    yield 2
    yield 3
    from itertools import count, chain
    yield from chain.from_iterable(
        zip(count(5, 6),
            count(7, 6)))

def primes(*, _known=PrimeCache([2, 3, 5, 7, 11, 13])):
    """Generate primes.
    
    Caches primes across generators.
    This may become inefficient for large
    """
    known = _known
    for c in candidate_primes():
        # skip already-checked numbers
        if known[-1] >= c:
            if c in known:
                yield c
            continue
        if all(c % p != 0 for p in known):
            # no known prime divides c
            known.append(c)
            yield c

def primerange(a, b=None):
    """Generate primes on [2, a) or a, b).

    See also sympy.ntheory.primerange.
    """
    if b is None:
        a, b = 2, a
    it = primes()
    while (p := next(it)) < a:
        continue
    yield p # first prime >= a
    while (p := next(it)) < b:
        yield p

def is_prime(n):
    # return n in primerange(n+1)
    if n == 2:
        return True
    bound = floor(sqrt(n))
    for p in primerange(bound+1):
        if n % p == 0:
            return False
    else:
        return True

def factorize(n):
    """Get prime factorization of n.

    Returns a Counter mapping factors to their
    exponent in the factorization.
      n == prod(p**k for p, k in factorize(n).items())

    No entry in the Counter will have a value of zero.
    """
    # Use a base Counter to avoid the extra
    # prime checks in Factors.__setitem__.
    factors = Counter()
    bound = floor(sqrt(n))
    for p in primerange(bound+1):
        # print("testing", p)
        while n % p == 0:
            factors[p] += 1
            n //= p
            # print(f"found {p}; reduced to {n}")
        if n == 1:
            break
    else: # we didn't reach n == 1
        factors[n] = 1

    return Factors(factors, _safe=True)

class Factors(Counter):
    """Counter that represents integer factors.

    Factors with zero occurences are removed from
    the mapping. Non-prime factors are broken down
    if added.

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
    """
    # Rejected:
    # Make composites return their factor counts, too.

    def __init__(self, *a, _safe=False, **k):
        super().__init__(*a, **k)
        if not _safe:
            for k in self:
                if not is_prime(k):
                    raise ValueError(f'factor {k} is not prime')

    @classmethod
    def of(cls, n):
        # really should be inlined
        return factorize(n)

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
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f'{cls}({prod}, {{{items}}})'

    def __repr__(self):
        cls = type(self).__name__
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f'{cls}({{{items}}})'

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
            return NotImplemente

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


def totient(n):
    if is_prime(n):
        return n - 1
    f = factorize(n)
    return prod(p**(k-1) * (p-1)
                for p, k in f.items())

fastexp = pow # add implementation later


# In[4]:


def primitive_root(p, *, smallest=False, base_tries=5, try_first=()):
    """Find a primitive root of p.
    
    Will try values in try_first first, then base_tries
    random values, then all values below p.

    The final, exhaustive search will be in random order
    for relatively small p, or ascending order for large p.

    If smallest is true, skip all other steps and only
    search in ascending order.
    """
    # if CHECK_PRIMES:
    #     assert is_prime(p)
    if p == 2:
        return 1
    elif p == 4:
        return 3

    # Reject all the numbers without primitive roots
    fs = factorize(p)
    if not (len(fs) == 1
            # above: prime^k, below: 2*prime^k
            or (len(fs) == 2 and fs[2] == 1)):
        raise ValueError(f'{p} has no primitive roots')

    if not is_prime(p):
        raise NotImplementedError(
            "primitive_root does not currently support non-primes")

    phi = p - 1
    factors = factorize(phi).keys()
    exponents = [phi//q for q in factors]
    def is_root(b):
        return all(fastexp(b, x, p) != 1
                   for x in exponents)
        # for x in exponents:
        #     if fastexp(b, x, p) == 1:
        #         return False
        # else:
        #     return True

    if smallest:
        for b in range(2, p-1):
            if is_root(b):
                return b
        assert False, f"unreachable: primitive roots mod {p} should exist"

    for b in try_first:
        if is_root(b):
            return b

    for _ in range(base_tries):
        b = random.randrange(2, p)
        if is_root(b):
            return b

    bs = range(2, p)
    if p <= 10_000:
        # Shuffle the items, but now it
        # actually takes space.
        bs = random.sample(bs, len(bs))
    for b in bs:
        if is_root(b):
            return b
    assert False, f"unreachable: primitive roots mod {p} should exist"

def all_primitive_roots(p):
    """Get all primitive roots of p.
    """
    raise NotImplementedError
    if CHECK_PRIMES:
        assert is_prime(p)
    if p == 2:
        return (1,)
    phi = p - 1
    factors = factorize(phi).keys()
    exponents = [phi//q for q in factors]
    roots = []
    for b in range(2, p):
        if all(fastexp(b, x, p) != 1
               for x in exponents):
            roots.append(b)
    return roots

def is_primitive_root(b, p, *, factors=None):
    """Check if b is a primitive root of p.

    If given, factors should be the unique
    prime factors of p - 1.
    """
    if CHECK_PRIMES:
        assert is_prime(p)
    if factors is None:
        phi = p - 1
        factors = factorize(phi).keys()
    else:
        phi = p
    exponents = [phi//q for q in factors]
    return all(fastexp(b, x, p) != 1
               for x in exponents)



# In[5]:

# Tests # TODO: negative test cases
def test_pr(n):
    assert sn.is_primitive_root(
        primitive_root(n), n), n
def poor_test():
  for p in primerange(1, 3000):
    test_pr(p)
# poor_test()


# In[7]:


# DH defs
from typing import NamedTuple

def alias(attr):
    @property
    def get(self):
        return getattr(self, attr)
    return get

def unalias(attr):
    @property
    def get(self):
        raise AttributeError(attr)
    return get

class Public(NamedTuple):
    p: int
    b: int
    bx: int
    # br = alias('bx')
    # bt = alias('bx')

class Private(NamedTuple):
    p: int
    b: int
    s: int
    # r = alias('s')
    # t = alias('s')

    def publish(self, cls=Public, /) -> Public:
        p, b, s = self
        # if isinstance(cls_or_name, str):
        #     raise TypeError("named publish not currently allowed")
        #     name = cls_or_name
        #     cls = NamedTuple(
        # b       "Public",
        #         [("p", int), ("b", int), ("b"+name, int)])
        return cls(p, b, fastexp(b, s, p))

    @classmethod
    def like(cls, pub, s):
        return cls(pub.p, pub.b, s)

    def encrypt(self, pub, m):
        assert self[:2] == pub[:2], (self, pub)
        p, b, s = self
        bxs = fastexp(pub.bx, s, p)
        return m * bxs % p

    def decrypt(self, pub, c):
        assert self[:2] == pub[:2], (self, pub)
        p, b, s = self
        bx_inv = fastexp(pub.bx, p-2, p)
        bxs_inv = fastexp(bx_inv, s, p)
        return bxs_inv * c % p

class PrivateR(Private):
    r = alias("s")
    def publish(self):
        return super().publish(PublicR)

class PrivateT(Private):
    t = alias("s")
    def publish(self):
        return super().publish(PublicT)

class PublicR(Public):
    br = alias("bx")

class PublicT(Public):
    bt = alias("bx")


# In[8]:


# PrivA = Private(p, b, r)
# PubA = Public(p, b, fastexp(b, r, p))
# PrivB = Private(p, b, t)
import builtins

def Alice(p=9511, b=2021, *, m=None):
    def print(*a):
        return
        builtins.print("Alice:", *a)
    m = m or 7654
    r = 2345
    key: Private = Private(p, b, r)
    print("key", key)
    print("mypub", key.publish())
    pub = yield key.publish()
    print("pub", pub)
    print("m", m)
    yield key.encrypt(pub, m)

def Bob(coms: Generator):
    def print(*a):
        return
        builtins.print("Bob:", *a)
    pub = next(coms)
    print("pub", pub)
    t = 6789
    key = Private.like(pub, t)
    print("key", key)
    print("mypub", key.publish())
    c = coms.send(key.publish())
    print("c", c)
    return key.decrypt(pub, c)

def listener(coms: Generator):
    def print(*a):
        builtins.print("mitm:", *a)
    try:
        adata = next(coms)
        while True:
            print("a->b:", adata)
            bdata = yield adata
            print("b->a:", bdata)
            adata = coms.send(bdata)
    except StopIteration:
        print('coms closed')

def mitm(sender):
    """Replaces the sendsr's message."""
    # communicates with a via the given
    # generator and with b as the generator
    def print(*a):
        builtins.print("mitm:", *a)
    r = 9821
    t = 1289

    apub = next(sender)

    fake_a = Private.like(apub, r)
    fake_b = Private.like(apub, t)

    bpub = yield fake_a.publish()
    c = sender.send(fake_b.publish())

    m = fake_b.decrypt(apub, c)
    print("secret message", m)
    fake_m = m * 2
    resp = yield fake_a.encrypt(bpub, fake_m)

Bob(mitm(Alice(m=1111)))


# In[9]:


# Mostly-manual version for debugging

# del p, b, r, t, x, br, bt, brt, btr
p = 29#9511
b = 3
assert is_primitive_root(b, p)
x = 18#7654
r = 5#2345
t = 12#6789
SA = priva = PrivateR(p, b, r)
PA = puba = priva.publish()
SB = privb = PrivateT.like(puba, t)
PB = pubb = privb.publish()

# Alice has: priva, pubb
# Bob has: privb, puba
print("PrivA", priva)
print("PrivB", privb)
print("PubA", puba)
print("PubB", pubb)

print("b^r", pow(b, r, p), "==", puba.br)
print("b^r^t =", brt := pow(pow(b, r, p), t, p))
print("b^t^r =", btr := pow(pow(b, t, p), r, p))

assert brt == pow(puba.br, privb.t, p)
assert btr == pow(pubb.bt, priva.r, p)
# Alice's side
c = priva.encrypt(pubb, x)
assert c == x * btr % p
# Bob's side
def decryptb(self, pub, c):
    p, b, t = self
    # b_inv = fastexp(b, p-2, p)
    # assert b_inv * b % p == 1, "b_inv"
    # bt_inv = fastexp(b_inv, t, p)
    # assert bt_inv * pow(b, t, p) % p == 1, "bt_inv"
    br_inv = fastexp(pub.br, p-2, p)
    assert pub.br * br_inv % p == 1, "br_inv"
    brt_inv = fastexp(br_inv, t, p)
    assert brt_inv * pow(pub.br, t, p) % p == 1, "btr_inv"
    return brt_inv * c % p
decryptb(privb, puba, c)


# In[2]:


from contextlib import contextmanager

@contextmanager
def modular_ints(n):
    ops = {}
    opnames = ("__add__",
               "__sub__",
               "__mul__")
    for op in opnames:
        ops[op] = opfn = getattr(int, op)
        def modop(a, b, *xs):
            return opfn(a, b, *xs)
        setattr(int, op, modop)
    try:
        yield
    finally:
        for op in opnames:
            setattr(int, op, ops[op])
