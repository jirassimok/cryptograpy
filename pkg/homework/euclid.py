# -*- flycheck-checker: python-pyright; -*-
from __future__ import annotations
import builtins
from collections import deque
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Final

from .util import (alternate_impl, copy_callable_type, cr, Verbosity,
                   is_verbose, printer, recursive_logging)

__all__ = [
    'euclid',
    'euclid_recursive',
    'ext_euclid',
    'ext_euclid_magic_index',
    'ext_euclid_full_table',
    'ext_euclid_full_columns'
]

## Extra setup

type IntOrTable = int | PseudoTable | deque[int]

# TODO: move to util module
MANUAL_DIVMOD: ContextVar[bool] = ContextVar('MANUAL_DIVMOD', default=False)
"""Whether to avoid reliance on // and % (and divmod).
"""

ALLOW_NEGATIVE: bool = False
"""Whether to allow negative inputs.

If MANUAL_DIVMOD is True, this must be false.
"""


@copy_callable_type(builtins.divmod)
@alternate_impl(MANUAL_DIVMOD, False, builtins.divmod)
def divmod(dividend: int, divisor: int, /) -> tuple[int, int]:
    # Manual divmod is not strictly equivalent to builtin divmod
    if dividend < 0 or divisor < 0:
        raise ValueError('divmod requires non-negative arguments')
    elif divisor == 0:
        raise ZeroDivisionError('divmod by zero')
    q, r = 0, dividend
    while r >= divisor:
        r -= divisor
        q += 1
    return q, r


def check_signs[**A, R](fn: Callable[A, R]) -> Callable[A, R]:
    """Decorate a function to make all int arguments respect ALLOW_NEGATIVE.

    Checking is done at call time on both positional and keyword arguments.
    """
    name = fn.__name__
    @wraps(fn)
    def sign_wrapper(*args: A.args, **kwargs: A.kwargs) -> R:
        if not ALLOW_NEGATIVE:
            if (any(arg < 0 for arg in args
                    if isinstance(arg, int))
                or any(val < 0 for val in kwargs.values()
                       if isinstance(val, int))):
                raise ValueError(f'{name}() arguments must be'
                                 ' non-negative (ALLOW_NEGATIVE)')
        return fn(*args, **kwargs)
    return sign_wrapper


## Euclidean Algorithm

@recursive_logging
@check_signs
def euclid_recursive(m: int, n: int, /) -> int:
    # # (sorting args not actually necessary)
    # if m < n:
    #     m, n = n, m
    if m == 0 or n == 0:
        return m or n
    q, r = divmod(m, n)
    if r == 0:
        return n
    return euclid_recursive(n, r)


@check_signs
def euclid(m: int, n: int, /, *, verbose:Verbosity=None) -> int:
    if is_verbose(verbose):
        if m == 0:
            def print_eqn():
                print(f'0 = {n} * 0 + {n}')
        elif n == 0:
            def print_eqn():
                print(f'{m} = 0 * 0 + {m}')
        else:
            # I wonder if I can narrow q at all...
            w_a = w_q = len(str(max(m, n)))
            w_b = len(str(n)) if m > n else w_a
            w_r = len(str(max(m, n) % min(m, n)))
            def print_eqn():
                print(f'{a:>{w_a}} = {b:>{w_b}} * {cr(q, w_q)} + {r:>{w_r}}')
    else:
        def print_eqn():
            pass

    if m == 0 or n == 0:
        print_eqn()
        return m or n

    # a = b * q + r

    b, r = m, n # just to initialize the loop nicely
    # # If we don't want to do the above line, we have this option:
    # a, b = m, n
    # q, r = divmod(a, b)
    # print_eqn()
    while r > 0:
        a, b = b, r
        q, r = divmod(a, b)
        print_eqn()

    return b


## Extended Euclidean Algorithm

@check_signs
def ext_euclid(m: int, n: int, /, *,
               verbose:Verbosity=None) -> tuple[int, int, int]:
    """Perform the Extended Euclidean Algorithm to find gcd and coefficients.

    Returns g, s, t such that s*m + t*n = g.

    Does not behave well with m or n equal to 0.
    """
    print_eqn = _ext_printer(m, n, verbose)

    s = deque([1, 0], maxlen=2)
    t = deque([0, 1], maxlen=2)

    # a = b * q + r
    b, r = m, n
    while r != 0:  # change this to >0 to restrict to positives
        a, b = b, r
        q, r = divmod(a, b)
        s.append(s[-2] - q * s[-1])
        t.append(t[-2] - q * t[-1])
        print_eqn(a, b, q, r, s, t)
    return b, s[-2], t[-2]

# Extremely busy function to prepare a nice printer for the extended algorithm.
def _ext_printer(m: int, n: int, verbose: Verbosity
                 ) -> Callable[[int, int, int, int,
                                IntOrTable, IntOrTable], None]:
    """Print table headers and create logging function for ext_euclid.
    """
    if is_verbose(verbose):
        return _verbose_ext_printer(verbose, m, n)
    else:
        def dummy_print_eqn(a, b, q, r, s, t):
            pass
        return dummy_print_eqn

def _verbose_ext_printer(verbose: Verbosity, m: int, n: int
                         ) -> Callable[[int, int, int, int,
                                        IntOrTable, IntOrTable], None]:
    builtins.print('verbose=', verbose)
    print = printer(is_verbose(verbose))

    # Short path for quick printing when an argument is zero.
    if m == 0 or n == 0:
        # Rather than rely on the algorithm to always print in this case, just
        # print the table right here.
        w_a, w_b = len(str(m)), len(str(n))
        print('Zero arg given: special-casing table')
        if n == 0: # args are (x, 0) -> s*x + t*0 = x
            print(f'{"a":^{w_a}} = b * q + {"r":^{w_a}} | s |  t  |')
            print(f'{m} = 0 * q + {m} | 1 | any |')
        else: # args are (0, x) -> s*0 + t*x = x
            print(f'a = {"b":^{w_b}} * q + r |  s  | t |')
            print(f'0 = {n} * 0 + 0 | any | 1 |')

        def print_nothing(*args, **kwargs):
            pass
        return print_nothing

    w_a = w_q = len(str(max(m, n)))
    w_b = len(str(n)) if m > n else w_a
    w_r = len(str(max(m, n) % min(m, n)))
    # if m < n, the first r is m, and it may be wider than the regular r width.
    # So in this case, right-align r in its regular column, then left-align the
    # result in a wider column.
    # (And by passing a different alignment function (c), we can make the
    #  header right-center above the "true" column while still right-padding.)
    def r_fmt(r, a: str | Callable[[str, int], str] = '>'):
        base = f"{r:{a}{w_r}}" if isinstance(a, str) else a(r, w_r)
        return f'{base:<{w_b}}' if m < n else base

    # add one for a negative sign (but don't give these columns a margin
    # below, so it doesn't look too spaced out)
    w_st = max(w_a, w_b) + 1

    def print_initial(i):
        a, b, q, r = ' ' * 4 if i else 'abqr'
        space = (w_st // 2) * ' '
        s, t = [space + 's', 1, 0], [space + 't', 0, 1]
        print(f'{cr(a, w_a)} = {cr(b, w_b)} * {cr(q, w_q)} + {r_fmt(r, cr)}'
              f' |{s[i]:>{w_st}} |{t[i]:>{w_st}} | ')
    print_initial(0)
    print_initial(1)
    print_initial(2)

    # Normalize s or t to int. Either int or last value from a table.
    def normalize(v: IntOrTable) -> int:
        if isinstance(v, int):
            return v
        elif isinstance(v, PseudoTable):
            return v[v.i-1]
        else:
            return v[-1]

    def print_eqn(a: int, b: int, q: int, r: int,
                  s: IntOrTable, t: IntOrTable) -> None:
        s, t = normalize(s), normalize(t)
        # if m < n, the first r is m, and it may be wider than the regular r
        # width. So in this case, right-align r in its regular column, then
        # left-align the result in a wider column.
        rx = f'{f"{r:>{w_r}}":<{w_b}}' if m < n else f'{r_fmt(r)}'
        print(f'{a:>{w_a}} = {b:>{w_b}} * {cr(q, w_q)} + {rx}'
              f' |{s:>{w_st}} |{t:>{w_st}} | ')
    return print_eqn


## Extended Euclidean Algorithm variant implementations

@check_signs
def ext_euclid_magic_index(m: int, n: int, /, *,
                           verbose:Verbosity=None) -> tuple[int, int, int]:
    """Variant that uses the PseudoTables to pretend its using a full table.
    """
    print_eqn = _ext_printer(m, n, verbose)

    # Swapping makes the output order less useful.
    # # Swap only because it looks nicer for the table's formatting.
    # if m < n:
    #     m, n = n, m

    s = PseudoTable([1, 0], maxlen=2)
    t = PseudoTable([0, 1], maxlen=2)
    # i: Final[int] = -1  # Not a real index.
    i: Final = PseudoTable.i

    # a = b * q + r
    b, r = m, n
    while r != 0:
        a, b = b, r
        q, r = divmod(a, b)
        s[i] = s[i-2] - q * s[i-1]
        t[i] = t[i-2] - q * t[i-1]
        print_eqn(a, b, q, r, s, t)

    return b, s[i-2], t[i-2]


@check_signs
def ext_euclid_full_table(m: int, n: int, /, *,
                          verbose:Verbosity=None) -> tuple[int, int, int]:
    """Variant that stores all intermediate variables in a full table.
    """
    print_eqn = _ext_printer(m, n, verbose)

    from typing import NamedTuple
    Row = NamedTuple('Row', [('a', int), ('b', int), ('q', int),
                             ('r', int), ('s', int), ('t', int)])

    table = [Row('-', '-', '-', '-', 1, 0), # type: ignore # pyright: ignore
             Row('-', m, '-', n, 0, 1)] # type: ignore # pyright: ignore
    while table[-1].r != 0:
        last = table[-1]
        a, b = last.b, last.r
        q, r = divmod(a, b)
        s = table[-2].s - q * table[-1].s
        t = table[-2].t - q * table[-1].t
        table.append(Row(a, b, q, r, s, t))
        print_eqn(*table[-1])
    return table[-1].b, table[-2].s, table[-2].t


@check_signs
def ext_euclid_full_columns(m: int, n: int, /, *,
                            verbose:Verbosity=None) -> tuple[int, int, int]:
    """Variant that stores all intermediate variables in columns.
    """
    print_eqn = _ext_printer(m, n, verbose)

    A: list[int] = ['-', '-'] # type: ignore # pyright: ignore
    B: list[int] = ['-', m]   # type: ignore # pyright: ignore
    Q: list[int] = ['-', '-'] # type: ignore # pyright: ignore
    R: list[int] = ['-', n]   # type: ignore # pyright: ignore
    S = [1, 0]
    T = [0, 1]

    while R[-1] != 0:
        a, b = B[-1], R[-1]
        q, r = divmod(a, b)
        s = S[-2] - q * S[-1]
        t = T[-2] - q * T[-1]
        new: int
        col: list[int]
        for new, col in ((a, A), (b, B), (q, Q), (r, R), (s, S), (t, T)):
            col.append(new)
        print_eqn(a, b, q, r, s, t)
    return B[-1], S[-2], T[-2]


## Excessive class structure for funny indexing tricks

type IntType[T] = Callable[[int], T]


class PseudoIndexMeta(type):
    """Metaclass for classes instantiable using + and -
    """
    def __new__(cls, name, bases, dict_):
        return super().__new__(cls, name, bases, dict_)
    def __add__[T](cls: Callable[[int], T], other: int) -> T:
        return cls(other) if isinstance(other, int) else NotImplemented
    def __sub__[T](cls: Callable[[int], T], other: int) -> T:
        return cls(-other) if isinstance(other, int) else NotImplemented


@dataclass(frozen=True)
class PseudoIndex(metaclass=PseudoIndexMeta):
    """Index representing the next row of a table.

    Subtracting from this type allows indexing earlier into the table.
    """
    offset: int

    def __post_init__(self):
        if self.offset >= 0:
            raise ValueError('Pseudoindex offset must be negative')

    def __repr__(self):
        return f'{type(self).__name__}{self.offset:+}'


class PseudoTable:
    """A weird view of the last rows of a table columns.

    Supports indexing via offsets from PseudoTable.i, and appending
    by assignment to index PseudoTable.i.
    """
    i: Final[type[PseudoIndex]] = PseudoIndex

    def __init__(self, iterable: Iterable[int], maxlen: int | None = None):
        self._data = deque(iterable, maxlen)

    def __repr__(self):
        # Use the deque's repr, but with this class' name.
        return repr(self._data).replace(type(self._data).__name__,
                                        type(self).__name__,
                                        1)

    def __getitem__(self, item: PseudoIndex) -> int:
        cls = type(self)
        if not isinstance(item, cls.i):
            raise ValueError(f'{cls.__name__} index must be an offset from '
                             f'{cls.__name__}.i')
        return self._data[item.offset]

    def __setitem__(self, item: type[PseudoIndex], value: int):
        if item is not self.i:
            name = type(self).__name__
            raise ValueError(f"{name} only supports setting index "
                             f"{name}.i, as an alias for 'append'")
        return self._data.append(value)
