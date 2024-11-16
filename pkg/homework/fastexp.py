# -*- flycheck-checker: python-mypy; -*-
# mypy: allow-redefinition
# Homework 3

from typing import Literal

from .util import is_verbose, Verbosity


## Implementation

def fastexp(base: int, exp: int, modulus: int | None = None,
            *, verbose: Verbosity = None) -> int:
    """Compute x**e.
    """
    if is_verbose(verbose):
        return verbose_fastexp(base, exp, modulus)

    if exp < 0:
        raise ValueError(f'exponent {exp} < 0')
    elif modulus == 0:
        raise ZeroDivisionError('modulus = 0')

    acc = 1
    while exp:
        if exp % 2:
            exp -= 1
            acc *= base
            if modulus is not None:
                acc %= modulus
        else:
            exp //= 2
            base **= 2
            if modulus is not None:
                base %= modulus
    return acc


def verbose_fastexp(base: int, exp: int, modulus: int | None = None) -> int:
    """Compute x**e.
    """
    if exp < 0:
        raise ValueError(f'exponent {exp} < 0')
    elif modulus == 0:
        raise ZeroDivisionError('modulus = 0')

    mw = len(str(modulus)) if modulus else None
    ew = len(str(exp))
    def print_sub(x, e, y, eqn=None):
        if mw is None:
            xw = len(str(base))
            yw = len(str(acc))
        else:
            xw = yw = mw
        parts = [f'| {x:>{xw}} | {e:>{ew}} | {y:>{yw}} |']
        if eqn is not None:
            parts.append(eqn)
        print(' '.join(parts))

    def print_row(mode: Literal['even', 'odd', 'both'] = 'both',
                  eqn: str | None = None):
        x, e, y = base, exp, acc
        if mode == 'both':
            print_sub(x, e, y)
        elif mode == 'even':
            print_sub(x, e, '.', eqn)
            # f'x <- x\N{superscript two} mod {modulus}')
        elif mode == 'odd':
            print_sub('.', e, y, eqn)
            # f'y <- x\N{multiplication sign}y mod {modulus}')

    acc = 1
    print_sub('x', 'e', 'y')
    print_row()
    while exp:
        if exp % 2:
            exp -= 1
            oldacc = acc
            acc *= base
            fullacc = acc
            if modulus is not None:
                acc %= modulus
            print_row('odd', f'≡ₘ {fullacc} = {base}×{oldacc}')
        else:
            exp //= 2
            oldbase = base
            base **= 2
            fullbase = base
            if modulus is not None:
                base %= modulus
            print_row('even', f'≡ₘ {fullbase} = {oldbase}²')
    return acc


def verbose2_fastexp(base: int, exp: int, modulus: int | None = None) -> int:
    """Extra-verbose version. Not important."""
    if exp < 0:
        raise ValueError(f'exponent {exp} < 0')
    elif modulus == 0:
        raise ZeroDivisionError('modulus = 0')

    acc = 1 # define early for use in printer (gets redefined later anyway)

    mw = len(str(modulus)) if modulus else None
    ew = len(str(exp))
    def print_sub(x, e, y, eqn=None, rhs=None):
        if mw is None:
            xw = len(str(base))
            yw = len(str(acc))
        else:
            xw = yw = mw
        parts = [f'| {x:>{xw}} | {e:>{ew}} | {y:>{yw}} |']
        if eqn is not None:
            parts.append(eqn)
        if rhs is not None:
            parts.append(rhs)
        print(' '.join(parts))

    def print_row(mode: Literal['even', 'odd', 'both'] = 'both',
                  eqn: str | None = None,
                  rhs: tuple[int, int] | None = None):
        x, e, y = base, exp, acc
        if eqn is None:
            eqn = ''

        rhs # use the variable so mypy can redefine it
        rhs: str | tuple[int, int] | None # add str
        if rhs is None:
            rhs = ''
        elif modulus is not None:
            rhs = f'{rhs[0]} \N{identical to} {rhs[1]} mod {modulus}'
        else:
            rhs = str(rhs[0])
        # In the two latter branches above, mypy thinks rhs[0] is (int | str),
        # but it works out in the end; now mypy knows rhs is str.

        if mode == 'both':
            print_sub(x, e, y)
        elif mode == 'even':
            print_sub(x, e, ' ', rhs, eqn)
            # f'x <- x\N{superscript two} mod {modulus}')
        elif mode == 'odd':
            print_sub(' ', e, y, rhs, eqn)
            # f'y <- x\N{multiplication sign}y mod {modulus}')

    print_sub('x', 'e', 'y')
    print_row()
    print_sub(' ', ' ', ' ')

    acc = 1
    while exp:
        if exp % 2:
            exp -= 1
            oldacc = acc
            acc *= base
            fullacc = acc
            if modulus is not None:
                acc %= modulus
            print_row('odd',
                      # f'y <- x\N{multiplication sign}y = '
                      f'{base}\N{multiplication sign}{oldacc} =',
                      (fullacc, acc))
        else:
            exp //= 2
            oldbase = base
            base **= 2
            fullbase = base
            if modulus is not None:
                base %= modulus
            print_row('even',
                      # f'x <- x\N{superscript two} = '
                      f'{oldbase}\N{superscript two} =',
                      (fullbase, base))
    return acc


## Tests
# no cover: start

def test_fastexp(fastexp_fn, test_cases):
    # Test cases: (test against builtins.pow in every case)

    def test(args, msg=None):
        realargs = [x for x in args if not isinstance(x, str)]
        assert fastexp_fn(*realargs) == pow(*realargs), (
            args if msg is None else f'{msg}: {args}')

    failures = []
    for n, args in enumerate(test_cases):
        try:
            test(args, msg=n)
        except AssertionError:
            failures.append((n, args))

    if failures:
        print('Tests failed:')
        for (n, args) in failures:
            print(f'\t{n}: {args}')
    else:
        print('All tests passed.')


def test_cases():
    from itertools import permutations
    from math import log10, floor

    def d(n): # number of digits
        return floor(log10(n)) + 1

    p3 = 127
    p60 = 529247038585542108568540290995084860068177551640322543102099
    p40 = 5365152086379702330152001265920318993223
    p30 = 491677946950317334566508347041
    return (
        ('initial case', p3, 329701361, p60),
        ('2 to a big power', 2, p60, p40),
        *((f'big primes p{d(a)}**p{d(b)} % p{d(c)}', a, b, c)
          for a, b, c in permutations((p30, p40, p60))),
        # that -> 351420569476919220033676836688505908233185988390868705944354
        ('non-mod large result', p3, 2575791), # result is 5418959 digits
    )


def run_tests():
    test_fastexp(fastexp, test_cases())

# no cover: stop


## Variants

def slowexp(base: int, exp: int, modulus: int | None = None):
    if exp < 0:
        raise ValueError(f'exponent {exp} < 0')
    elif modulus == 0:
        raise ZeroDivisionError('modulus = 0')

    acc = 1
    for i in range(exp):
        acc *= base
        if modulus is not None:
            acc %= modulus
    return acc


def fastexp_recursive(base: int,
                      exp: int,
                      mod: int | None = None):
    if exp < 0:
        raise ValueError(f'exponent {exp} < 0')
    elif mod == 0:
        raise ZeroDivisionError('modulus = 0')

    def fastexp(base: int, exp: int, mod: int | None = None,
                *, acc: int = 1) -> int:
        if mod is None:
            if exp == 0:
                return acc
            elif exp % 2:
                return fastexp(base, exp - 1, mod, acc=(acc * base))
            else:
                return fastexp(base**2, exp // 2, mod, acc=acc)
        else:
            if exp == 0:
                return acc
            elif exp % 2:
                return fastexp(base, exp - 1, mod, acc=(acc*base % mod))
            else:
                return fastexp(base**2 % mod, exp // 2, mod, acc=acc)

    return fastexp(base, exp, mod)


# Non-branching version (well, it still branches for the modulus)
def fastexp2(x: int, exp: int, m: int | None = None) -> int:
    """
    m must be positive.
    """
    if exp < 0:
        raise ValueError(f'exponent {exp} < 0')
    elif m == 0:
        raise ZeroDivisionError('modulus = 0')

    e0 = exp % 2
    exp = (exp - e0) // 2
    acc = x ** e0

    while exp:
        e0 = exp % 2
        exp = (exp - e0) // 2
        x = x ** 2
        acc *= x ** e0
        if m:
            x %= m
            acc %= m

    return acc
