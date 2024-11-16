# -*- flycheck-checker: python-mypy; -*-
# Miller-Rabin primality test
from __future__ import annotations
from collections.abc import Iterable

from .fastexp import fastexp as _fastexp
from .util import is_verbose, printer, substr, supstr, Verbosity

# Per Wikipedia, there is no prime below 2**64 that is strongly pseudoprime to
# all these bases:
# 2, 325, 9375, 28178, 450775, 9780504, 1795265022


def fastexp(base: int, exp: int, modulus: int):
    """fastexp, but returns -1 instead of modulus-1
    """
    r = _fastexp(base, exp, modulus, verbose=False)
    if r == modulus - 1:
        return -1
    else:
        return r


def test_prime_base(n: int, /, *, m: int, r: int, b: int,
                     verbose: Verbosity = None) -> bool:
    print = printer(is_verbose(verbose))

    bs = fastexp(b, m, n)
    print(f'b₀ = {bs} = {b}{supstr(m)} % {n}')

    if bs == 1 or bs == -1:
        return True

    for s in range(1, r):
        last = bs
        bs = fastexp(bs, 2, n)
        print(s)
        print(f'b{substr(s)} = {bs} = {last}² % {n}')
        if bs == -1:
            return True
        elif bs == 1:
            return False
    else:
        return False


def strong_prime_test(n: int, bases: Iterable[int], /,
                      *, verbose: Verbosity = None):
    print = printer(is_verbose(verbose))

    if n < 2:
        return False
    elif n == 2:
        return True
    elif n % 2 == 0:
        return False

    m = n - 1
    r = 0
    while m % 2 == 0:
        r += 1
        m //= 2

    print(f'n = {n} = 2^{r} * {m}, bases={bases}')
    for b in bases:
        if not test_prime_base(n, m=m, r=r, b=b, verbose=verbose):
            print('not prime to base', b)
            return False
        print('prime to base', b)
    else:
        return True


# def check(n, b):
#     r, m = _get_rm(n)
#     return check_prime_base(n=n, m=m, r=r, b=b)
