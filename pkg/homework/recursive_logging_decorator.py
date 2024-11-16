# -*- flycheck-checker: python-pyright; -*-

# Drafting for / old versions of the recursive logging decorator's typing.

# Ignore bad return type from __new__:
# mypy: disable-error-code=misc
import builtins
from functools import wraps
from typing import Callable, overload, Self, TYPE_CHECKING

from typing_extensions import TypeIs

# class CallableLikeMeta(type):
#     def __subclasscheck__(cls, sub, /):
#         return issubclass(sub, Callable)
#
#     def __instancecheck__(cls, obj, /):
#         return isinstance(obj, Callable)
#
#     # Runtime only; type checkers use the generic specification instead
#     def __getitem__(cls, type_params, /): # -> TypeForm[Callable]
#         return Callable[type_params]  # pyright: ignore[reportInvalidTypeForm]
#
# class callable2[**Params, R](metaclass=CallableLikeMeta):
#     # Trying to override __new__ to act as a function and still get
#     # good results is wild.
#     #
#     # Pyright allows replacing __new__'s return type incompatibly,
#     # while mypy does not.
#     #
#     # Mypy treats the output of a class call as an instance.
#     # Pyright treats the output of a class call as the output of __new__.
#
#     # So basically, this works with Pyright, but not Mypy.
#
#     # The next line checks with pyright, but the mypy ignore comment disables pyright, too.
#     def __new__(cls, obj, /) -> TypeIs[Callable[..., object]]: # type: ignore[misc]
#         return builtins.callable(obj)
#
# The new TypeIs and TypeExpr may be useful, but they aren't supported for my
# Python version.

type Decorator[F: Callable] = Callable[[F], F]

# The problem is visible in this definition:
# def fn[T]() -> T: ...
# But the same error isn't detected for this one:
# def fn[T]() -> list[T]: ...

@overload
def recursive_logging(*argnames: str) -> Decorator:
    ...
@overload
def recursive_logging[F: Callable](fn: F, /) -> F:
    ...
def recursive_logging[F: Callable](arg: F | str | None = None, /, *argnames: str) -> F | Decorator:
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
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                fmt = ', '.join(f'{n}={a}' for n, a in zip(argnames, args))
                if kwargs:
                    print(f'({fmt})', kwargs, sep=', ')
                else:
                    print(fmt)
                return fn(*args, **kwargs)
            return wrapper
        return decorator
    elif argnames:
        # direct call with full arg; not allowed
        raise TypeError('recursive_logging must be used as a decorator')
    else:
        @wraps(arg)
        def wrapper(*args, **kwargs):
            if kwargs:
                print(args, kwargs, sep=', ')
            else:
                print(args)
            return arg(*args, **kwargs)
        return wrapper


class recursive_log_args:
    # This needs an ignore[misc] from Mypy to be valid (and still can't be used
    # directly).
    @overload
    def __new__[**Args, R](cls, fn: Callable[Args, R], /) -> Callable[Args, R]:
        ...
    @overload
    def __new__(cls, /, *argnames: str) -> Self:
        ...
    def __new__[**Args, R](cls, arg: Callable[Args, R] | str | None = None, /, *argnames: str
                           ) -> Self | Callable[Args, R]:
        if arg is None or isinstance(arg, str):
            return super().__new__(cls)
        elif argnames:
            # direct call with full arg; not allowed
            raise TypeError('recursive_logging must be used as a decorator')
        else:
            return cls()(arg)

    def __init__(self, *argnames: str):
        if argnames and argnames[0] is None:
            argnames = argnames[1:]
        self.argnames = tuple(argnames)

    def __call__[**Args, R](self, fn: Callable[Args, R]) -> Callable[Args, R]:
        argnames = self.argnames
        if argnames:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                fmt = ', '.join(f'{n}={a}' for n, a in zip(argnames, args))
                if kwargs:
                    print(f'({fmt})', kwargs, sep=', ')
                else:
                    print(fmt)
                return fn(*args, **kwargs)
        else:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                if kwargs:
                    print(args, kwargs, sep=', ')
                else:
                    print(args)
                return fn(*args, **kwargs)
        return wrapper

# To make the above compatible with Mypy, you can use something like this:


if TYPE_CHECKING:
    from typing import Protocol

    class SimpleDecorator(Protocol):
        def __call__[**Args, R](self, fn: Callable[Args, R], /) -> Callable[Args, R]:
            ...

    @overload
    def recursive_log_args2(*argnames: str) -> Decorator:
        ...
    @overload
    def recursive_log_args2[F: Callable](fn: F, /) -> F:
        ...
    def recursive_log_args2[F: Callable](arg: F | str | None = None, /, *argnames: str) -> F | Decorator:
        ...
