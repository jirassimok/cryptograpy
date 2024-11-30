# -*- flycheck-checker: python-pyright; -*-
from __future__ import annotations
from collections.abc import Iterable
from contextvars import ContextVar
from functools import wraps
from itertools import takewhile, dropwhile
from operator import itemgetter
from typing import (Any, Callable, cast, Never, overload, Protocol,
                    TYPE_CHECKING)


## Settings

# These should be contextvars or something else importable.

VERBOSE: bool = False

USE_SYMPY: ContextVar[bool] = ContextVar('USE_SYMPY_PRIMES', default=False)
"""Whether to use sympy functions for various underlying operations.
"""

## Typing helpers

def copy_callable_type[**A, R](f: Callable[A, R]) -> Callable[[Callable],
                                                              Callable[A, R]]:
    """Decorate a function to have the argument's type.
    """
    # Making "f: Callable" makes Pyright unhappy with overloaded args.
    return lambda fn: fn

def copy_args[**A, R, R2](f: Callable[A, R]) -> Callable[[Callable[..., R2]],
                                                         Callable[A, R2]]:
    """Decorate a function to copy the argument's parameter types.
    """
    return lambda fn: fn

def returning_type_of[F: Callable, **A](f: F) -> Callable[[Callable[A, Any]],
                                                          Callable[A, F]]:
    """Decorate a function to have the argument's type as its return type.

    Allows type-checking of calls, but not of the definition.
    """
    return lambda fn: fn


## Alternative implementation helper

@overload
def alternate_impl(option: ContextVar[bool], impl: Callable,
                   /) -> SimpleDecorator:
    ...
@overload
def alternate_impl(option: ContextVar[bool], val: bool, impl: Callable,
                   /) -> SimpleDecorator:
    ...
def alternate_impl(option: ContextVar[bool], val: bool | Callable,
                   impl: Callable = cast(Callable, None),
                   /) -> SimpleDecorator:
    """Make a function use an alternate implementation in certain contexts.

    When the option has the given value (default True), use impl instead of
    the decorated function.
    """
    if not isinstance(val, bool):
        # I want these assignments in the other order, but Mypy has a bug.
        impl, val = val, True
    def decorator(fn):
        @wraps(fn)
        def alt_wrapper(*args, **kwargs):
            if option.get() == val:
                return impl(*args, **kwargs)
            else:
                return fn(*args, **kwargs)
        return alt_wrapper
    return decorator


## Logging utilities

# TODO: Just use a normal logging system instead.

type Verbosity = bool | None | _use_is_verbose

if TYPE_CHECKING:
    # This forces anyone using Verbosity as the type
    # to use is_verbose. This is silly.
    class _use_is_verbose:
        """Indicates the default verbosity.
        """
        def __bool__(self) -> Never:
            raise TypeError

    class CheckedVerbosity:
        pass

    def is_verbose(verbose: Verbosity) -> CheckedVerbosity:
        ...
else:
    type _use_is_verbose = None
    type CheckedVerbosity = Verbosity

    def is_verbose(verbose: Verbosity) -> bool:
        """Converts None to default verbosity setting.
        """
        return verbose or verbose is None and VERBOSE


# TODO (?) Make verbose=IOBase write to that (makes it testable)
@returning_type_of(print)
def printer(verbose: CheckedVerbosity):
    """Get a print function.

    Given None, consult module VERBOSE setting.
    """
    if verbose or (verbose is None and VERBOSE):
        return print
    else:
        def silent_print(*args, **kwargs):
            pass
        return silent_print

def cr(s, width):
    """Center-format with right bias. For use in f-strings.

    For odd widths, centers normally.
    For even widths, centers in the width-1 right columns.
    """
    if width % 2:
        return f'{s:^{width}}'
    else:
        return f'{f"{s:^{width - 1}}":>{width}}'


## Additional formatting helpers

def supstr(n: int) -> str:
    """Get a unicode superscript version of a base-10 number.
    """
    if n == 0:
        return '⁰'
    digits = []
    while n:
        digits.append(n % 10)
        n //= 10
    return ''.join(itemgetter(*reversed(digits))('⁰¹²³⁴⁵⁶⁷⁸⁹'))


def substr(n: int) -> str:
    """Get a unicode subscript version of a base-10 number.
    """
    if n == 0:
        return '₀'
    digits = []
    while n:
        digit = n % 10
        n //= 10
        digits.append(chr(0x2080 + digit))
    return ''.join(reversed(digits))


## Iterator helpers

def takebetween[T](iterable: Iterable[T], start, stop) -> Iterable[T]:
    """Get values in the range [start, stop) from an iterator.

    Gets the first run of values in the range if unsorted.
    """
    return takewhile(lambda x: x < stop,
                     dropwhile(lambda x: x < start, iterable))


## Recursive function logging decorator

class SimpleDecorator(Protocol):
    def __call__[**Args, R](self, fn: Callable[Args, R],
                            /) -> Callable[Args, R]:
        ...

@overload
def recursive_logging(*argnames: str) -> SimpleDecorator:
    ...
@overload
def recursive_logging[**Args, R](fn: Callable[Args, R],
                                 /) -> Callable[Args, R]:
    ...
def recursive_logging[**Args, R](arg: Callable[Args, R] | str | None = None,
                                 /, *argnames: str
                                 ) -> Callable[Args, R] | SimpleDecorator:
    """Decorator that makes a recursive function log its arguments.

    Applied directly to a function, prints the functions arguments
    before each call.

    Given arguments, produces a decorator that prints those arguments as
    the names of the decorated function's arguments.

    Prints arguments as given, with keyword arguments separate from
    positional arguments.
    """
    if arg is None:
        return recursive_logging
    elif isinstance(arg, str):
        argnames = (arg, *argnames)
        def decorator[**DA, DR](fn: Callable[DA, DR]) -> Callable[DA, DR]:
            @wraps(fn)
            def rec_wrapper(*args: DA.args, **kwargs: DA.kwargs) -> DR:
                # If not verbose, don't bother formatting output.
                if not VERBOSE:
                    return fn(*args, **kwargs)
                fmt = ', '.join(f'{n}={a}' for n, a in zip(argnames, args))
                if kwargs:
                    print(f'({fmt})', kwargs, sep=', ')
                else:
                    print(fmt)
                return fn(*args, **kwargs)
            return rec_wrapper
        return decorator
    elif argnames:
        # direct call with full arg; not allowed
        raise TypeError('recursive_logging must be used as a decorator')
    else:
        @wraps(arg)
        def rec_wrapper(*args: Args.args, **kwargs: Args.kwargs) -> R:
            if not VERBOSE:
                pass
            elif kwargs:
                print(args, kwargs, sep=', ')
            else:
                print(args)
            return arg(*args, **kwargs)
        return rec_wrapper
