from __future__ import annotations

from collections.abc import Buffer
from dataclasses import dataclass
import functools
import numbers
import operator
import types
from typing import (Callable, Concatenate, Literal, Never, overload, Protocol,
                    Self, SupportsIndex, SupportsInt, TYPE_CHECKING)


## Number-container numbers.Integral implementation

@dataclass(frozen=True)
class Integral:
    """Like int, but has some number-theoretic operations.
    """
    value: int

    # It would be better to use the number module docs' recommended approach of
    # checking for subtypes in the operator methods, or passing an operator
    # method into a generic op-wrapper function, to avoid the checks here.
    #
    # The solution for subclasses here is only suitable as long as subclasses
    # don't change the operations at all.
    @classmethod
    def _wrap[Sub: Self](cls, value: int | Sub) -> Self:
        if isinstance(value, int):
            return cls(value)
        elif isinstance(value, cls):
            # If attempting to wrap a value that we've already wrapped, don't.
            return value
        elif isinstance(value, Integral):
            # I don't like using Integral directly here, but it is slightly
            # nicer than my other idea (reverse subclass check).
            #
            # If the value is an Integral and not our subclass, assume it is
            # our superclass and pull its value into the more specific class.
            if issubclass(cls, type(value)):
                return cls(value.value)
            else:
                # Shouldn't be reachable as long as no more child classes are
                # created.
                raise TypeError("Can't combine unrealted Integral subclasses")
        else:
            # We somehow got a non-int, non-Integral input.
            raise TypeError(
                f'{cls.__name__} can only wrap ints and other Integrals')

    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return f'{type(self).__name__}({self.value})'

    # Technically, all methods should have all-positional-only arguments, but
    # Pyright defines Integral with keyword-optional parameters, so that fails
    # to type-check correctly.

    def divides(self, other: int | Self, /) -> bool:
        return other // self.value * self.value == other
    def divisible_by(self, other: int | Self, /) -> bool:
        return self.value // other * other == self.value

    def __or__(self, other: int | Self) -> bool:
        return self.divides(other)

    def __ror__(self, other: int | Self) -> bool:
        return self.divisible_by(other)

    # Operations required for numbers.Integral
    def __index__(self) -> int:
        return self.value

    def __int__(self) -> int:
        return self.value

    def __pos__(self) -> Self:
        return self._wrap(+self.value)
    def __neg__(self) -> Self:
        return self._wrap(-self.value)

    def __abs__(self) -> Self:
        return self._wrap(abs(self.value))

    def __ceil__(self) -> Self:
        return self
    def __floor__(self) -> Self:
        return self
    def __trunc__(self) -> Self:
        return self
    def __round__(self, ndigits: int | None = None, /) -> Self:
        return self

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        return self.value == other
    def __lt__(self, other) -> bool:
        return self.value < other
    def __le__(self, other) -> bool:
        return self.value <= other
    def __gt__(self, other) -> bool:
        return self.value > other
    def __ge__(self, other) -> bool:
        return self.value >= other

    # Arithmetic operations

    def __add__(self, other: int | Self) -> Self:
        return self._wrap(self.value + other)
    def __radd__(self, other: int | Self) -> Self:
        return self._wrap(other + self.value)

    def __sub__(self, other: int | Self) -> Self:
        return self._wrap(self.value - other)
    def __rsub__(self, other: int | Self) -> Self:
        return self._wrap(other - self.value)

    def __mul__(self, other: int | Self) -> Self:
        return self._wrap(self.value * other)
    def __rmul__(self, other: int | Self) -> Self:
        return self._wrap(other * other)

    def __truediv__(self, other: int | Self) -> Never:
        raise TypeError(f'Can not true-divide {type(self).__name__}')
    def __rtruediv__(self, other: int | Self) -> Never:
        raise TypeError(f'Can not true-divide by {type(self).__name__}')
    # Also skipping __matmul__, __rmatmul__

    def __floordiv__(self, other: int | Self) -> Self:
        return self._wrap(self.value // other)
    def __rfloordiv__(self, other: int | Self) -> Self:
        return self._wrap(other // self.value)

    def __mod__(self, other: int | Self) -> Self:
        return self._wrap(self.value % other)
    def __rmod__(self, other: int | Self) -> Self:
        return self._wrap(other % self.value)

    def __divmod__(self, other: int | Self) -> tuple[Self, Self]:
        q, r = divmod(self.value, other)
        return self._wrap(q), self._wrap(r)
    def __rdivmod__(self, other: int | Self) -> tuple[Self, Self]:
        q, r = divmod(other, self.value)
        return self._wrap(q), self._wrap(r)

    # int.__pow___ has a very fancy signature
    def __pow__(self, exponent: int | Self,
                mod: int | Self | None = None) -> Self:
        if mod is not None:
            mod = int(mod)
        return self._wrap(pow(self.value, int(exponent), mod))

    def __rpow__(self, base: int | Self,
                 mod: int | Self | None = None) -> Self:
        if mod is not None:
            mod = int(mod)
        return self._wrap(pow(int(base), self.value, mod))

    # Ban bitwise operations

    @classmethod
    def _bitwise_op(cls) -> Never:
        raise TypeError(f'Can not operate bitwise on {cls.__name__}')

    def __invert__(self) -> Never:
        self._bitwise_op()

    def __lshift__(self, other: int | Self) -> Never:
        self._bitwise_op()
    def __rlshift__(self, other: int | Self) -> Never:
        self._bitwise_op()

    def __rshift__(self, other: int | Self) -> Never:
        self._bitwise_op()
    def __rrshift__(self, other: int | Self) -> Never:
        self._bitwise_op()

    def __and__(self, other: int | Self) -> Never:
        self._bitwise_op()
    def __rand__(self, other: int | Self) -> Never:
        self._bitwise_op()

    def __xor__(self, other: int | Self) -> Never:
        self._bitwise_op()
    def __rxor__(self, other: int | Self) -> Never:
        self._bitwise_op()


class TrueIntegral(Integral, numbers.Integral): # type: ignore[misc]
    pass


## True int implementation

# This is defined in the typeshed rather than typing for some reason.
class SupportsTrunc(Protocol):
    def __trunc__(self) -> int: ...

class SupportsWrapping[In, Out](Protocol):
    """Protocol for types that can be constructed from an existing instance.
    """
    __name__: str
    def __call__(self, wrapped: In, /) -> Out:
        ...
    def mro(self) -> list[type]:
        ...

type Method[T, **A, R] = Callable[Concatenate[T, A], R]


### The descriptor that automates method implementation

class WrappedOperator[Sub, Base, **Args]:
    # Sub is the wrapping type
    # Base is the base type
    # Args is the params for the method, excluding the self param

    _wrapped: Method[Base, Args, Base] | None
    """The wrapped method.

    This is set either in the constructor or by a metaclass.
    Should be an unbound instance method.
    """
    _owner: SupportsWrapping[Base, Sub]
    __name__: str

    def __init__(self, wrapping: Method[Base, Args, Base] | None = None):
        """
        wrapping: method being wrapped
        """
        self._wrapped = wrapping
        self._wrap_as = None

    @property
    def owner(self) -> SupportsWrapping[Base, Sub]:
        return self._owner

    @property
    def name(self) -> str:
        return self.__name__

    def __repr__(self):
        # Like function repr, but doesn't include args
        return f'<function {self.__module__}.{self.__qualname__}>'

    def __set_name__(self, owner: SupportsWrapping[Base, Sub], name: str):
        if self._wrapped is None:
            for cls in owner.mro()[1:]:
                if hasattr(cls, name):
                    attr = getattr(cls, name)
                    if not callable(attr):
                        raise TypeError(
                            f'wrapped method {name} is not callable in {cls}')
                    self._wrapped = attr
                    break
            else:
                raise TypeError(
                   f'could not find super method {name} for {owner}')
        assert self._wrapped is not None # for typing purposes
        functools.update_wrapper(self, self._wrapped)
        self._owner = owner

    def __call__(self, instance: Base, /,
                 *a: Args.args, **k: Args.kwargs) -> Sub:
        if self._wrapped is None:
            raise ValueError(
                f'{type(self).__name__} missing wrapped method')
        return self.owner(self._wrapped(instance, *a, **k))


    @overload
    def __get__(self, obj: None, objtype=None) -> Self:
        ...
    @overload
    def __get__(self, obj: Base,
                objtype=None) -> Method[Base, Args, Sub]:
        ...
    def __get__(self, obj: Base | None,
                objtype=None) -> Self | Method[Base, Args, Sub]:
        if obj is None:
            return self
        return types.MethodType(self, obj)


class BannedOperator[Base, **Args](WrappedOperator[Never, Base, Args]):
    def __init__(self, opname, fmt='{cls} does not support {opname}'):
        super().__init__()
        self._message = (fmt, opname)

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        fmt, op = self._message
        self._message = fmt.format(cls=self.owner.__name__, opname=op)

    def __call__(self, instance: Base, /,
                 *args: Args.args, **kwargs: Args.kwargs) -> Never:
        raise TypeError(self._message)

from typing import Any, cast

def wrapped[T: WrappedOperator[Any, Any, Any]]() -> T: # type: ignore[type-var,misc] # pyright: ignore[reportInvalidTypeVarUse]
    return WrappedOperator()  # type: ignore[return-value] # pyright: ignore[reportReturnType]


### Special base int class for typing purposes

if TYPE_CHECKING:
    _PositiveInt = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    _NegativeInt = Literal[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10,
                           -11, -12, -13, -14, -15, -16, -17, -18, -19, -20]


class BaseInt(int):
    """Subclass of int that fixes type hinting for a handful of methods.
    """
    if TYPE_CHECKING:
        # Mark 'self' as positional-only for all of these
        def as_integer_ratio(self, /) -> tuple[int, Literal[1]]: ...
        def conjugate(self, /) -> int: ...
        def bit_length(self, /) -> int: ...
        def bit_count(self, /) -> int: ...
        def to_bytes(self, /, length: SupportsIndex = 1,
                     byteorder: Literal["little", "big"] = "big",
                     *, signed: bool = False) -> bytes: ...
        def is_integer(self, /) -> Literal[True]: ...
        def __neg__(self, /) -> int: ...
        def __pos__(self, /) -> int: ...
        def __invert__(self, /) -> int: ...
        def __trunc__(self, /) -> int: ...
        def __ceil__(self, /) -> int: ...
        def __floor__(self, /) -> int: ...
        def __getnewargs__(self, /) -> tuple[int]: ...
        def __float__(self, /) -> float: ...
        def __int__(self, /) -> int: ...
        def __abs__(self, /) -> int: ...
        def __hash__(self, /) -> int: ...
        def __bool__(self, /) -> bool: ...
        def __index__(self, /) -> int: ...
        # These aren't really @properties (they're types.GetSetDescriptorType),
        # so the keyword-only argument doesn't matter, because it's not
        # accessible anyway.
        # @property
        # def real(self, /) -> int: ...
        # @property
        # def imag(self, /) -> Literal[0]: ...
        # @property
        # def numerator(self, /) -> int: ...
        # @property
        # def denominator(self, /) -> Literal[1]: ...

        # Overloading __pow__ normally is nigh impossible because of the
        # extremely-specific overloads provided by int.
        #
        # And for some reason, even if you copy the typeshed definition for
        # int.__pow__, Pyright thinks the overloads are in the wrong order.
        #
        # This is basically the same as int.__pow__, but returning int instead
        # of Literal[1] for the 0 case.
        @overload # type: ignore[override]
        def __pow__(self, x: Literal[0], mod: None = None, /) -> int: ...
        @overload
        def __pow__(self, x: _PositiveInt, mod: None = None, /) -> int: ...
        @overload
        def __pow__(self, x: _NegativeInt, mod: None = None, /) -> float: ...
        @overload
        def __pow__(self, x: int, mod: None = None, /) -> Any: ...
        @overload
        def __pow__(self, x: int, mod: int, /) -> int: ...
        def __pow__( # pyright: ignore[reportIncompatibleMethodOverride,reportInconsistentOverload]
                self, x, mod=None, /):
            ...


### The true Integer subclass

class Integer(BaseInt):
    """Like int, but has some number-theoretic operations.
    """
    @overload
    def __new__(cls, x: (str | Buffer | SupportsIndex
                         | SupportsInt | SupportsTrunc) = 0, /) -> Self: ...
    @overload
    def __new__(cls, x: str | bytes | bytearray, /,
                base: SupportsIndex = 10) -> Self:
        ...
    def __new__(cls, x = 0, /, base = None): # pyright: ignore[reportInconsistentOverload]
        if base is None:
            return super().__new__(cls, x)
        else:
            return super().__new__(cls, x, base) # pyright: ignore

    def __str__(self) -> str: # str as bare int
        return super().__repr__()

    def __repr__(self) -> str: # repr as Integer(value)
        return f'{type(self).__name__}({super().__repr__()})'

    def divides(self, other: int, /) -> bool:
        return other // self * self == other

    def divisible_by(self, other: int, /) -> bool:
        return self // other * other == self

    def __or__(self, other: int) -> bool:
        return self.divides(other)

    # Pyright will not handle this "correctly," and will report int
    # for reveal_type(int() | Integer()). See issues 3368 and 3862.
    #
    # If both operands to an operator are, for example, ints, you can't tell
    # statically if they are ints or subclasses thereof, so you can't use a
    # subtype's operator. Mypy handles this specifically for the case where one
    # operands is known to be a subclass of the other, but Pyright will not.
    def __ror__(self, other: int) -> bool:
        return self.divisible_by(other)

    # Assigning directly to WrappedOperator works in pyright
    # __add__ = WrappedOperator()
    # With explicit type parameters, it works in mypy:
    # __add__ = WrappedOperator[Self, int, [int]]()
    #
    # But I want something shorter that works in both, so I define various
    # local subclasses to handle that.

    # I want to mark these as type_check_only, but then Pyright complains
    # that I'm actually using them.

    # Pyright doesn't like using Self here, but it seems to work anyway
    class UnOp(WrappedOperator[Self, int, []]): # pyright: ignore
        pass

    class BinOp(WrappedOperator[Self, int, [int]]): # pyright: ignore
        pass

    class BitOp(BannedOperator[int, [int]]): # pyright: ignore
        def __init__(self):
            super().__init__('bitwise operations')

    # These aren't overridden:
    # - __index__, __hash__, __matmul__, __rmatmul__,
    # - Type conversions (int, bool, float)
    # - All boolean comparison operators (eq, ne, gt, ge, lt, le)

    __pos__: Callable[[int], int] = wrapped()
    __neg__ = UnOp()
    __abs__ = UnOp()

    __ceil__ = UnOp()
    __floor__ = UnOp()
    __trunc__ = UnOp()

    __add__ = BinOp()
    __radd__ = BinOp()
    __sub__ = BinOp()
    __rsub__ = BinOp()
    __mul__ = BinOp()
    __rmul__ = BinOp()
    __floordiv__ = BinOp()
    __rfloordiv__ = BinOp()
    __mod__ = BinOp()
    __rmod__ = BinOp()

    __truediv__ = BannedOperator[int, [int]]('/')
    __rtruediv__ = BannedOperator[int, [int]]('/')

    __invert__ = BannedOperator[int, []]('bitwise operations')
    __lshift__ = BitOp()
    __rlshift__ = BitOp()
    __rshift__ = BitOp()
    __rrshift__ = BitOp()
    __and__ = BitOp()
    __rand__ = BitOp()
    __xor__ = BitOp()
    __rxor__ = BitOp()

    del UnOp, BinOp, BitOp

    # Implementation for divmod is unique because of the tuple return
    def __divmod__(self, other: int) -> tuple[Self, Self]:
        q, r = super().__divmod__(other)
        return type(self)(q), type(self)(r)
    def __rdivmod__(self, other: int) -> tuple[Self, Self]:
        q, r = super().__rdivmod__(other)
        return type(self)(q), type(self)(r)

    @overload # type: ignore[override]
    def __pow__(self, x: Literal[0], mod: None = None, /) -> Self: ...
    @overload
    def __pow__(self, x: _PositiveInt, mod: None = None, /) -> Self: ...
    @overload
    def __pow__(self, x: _NegativeInt, mod: None = None, /) -> float: ...
    @overload
    def __pow__(self, x: int, mod: None = None, /) -> Any: ...
    @overload
    def __pow__(self, x: int, mod: int, /) -> Self:
        ...
    def __pow__( # pyright: ignore[reportIncompatibleMethodOverride]
            self, exponent: int, mod: int | None = None, /) -> Self | float:
        return type(self)(super().__pow__(exponent, mod))

    # Modified behavior
    # I'm not sure why Mypy complains that this "unsafely overlaps" with
    # int.__pow__.
    def __rpow__(self, other: int, # type: ignore[misc]
                 mod: int | None = None, /) -> int:
        if mod is None and self < 0:
            raise ValueError(f'can not use negative {type(self).__name__}'
                             ' as non-modular exponent')
        return type(self)(super().__rpow__(other, mod))


## Bit class

@dataclass(frozen=True)
class Bit(numbers.Integral):
    """A single bit that acts like an integer.

    Most operators return bits regardless of the other argument's type
    (though a few do not, notably when the bit is the right-hand argument).
    """
    value: int

    def __post_init__(self):
        object.__setattr__(self, 'value', self.value % 2)

    def __repr__(self):
        return f'{type(self).__name__}({self.value})'

    @staticmethod
    def _operator(opfn):
        def op(self, other):
            cls = type(self)
            if isinstance(other, cls):
                return cls(opfn(self.value, other.value))
            elif isinstance(other, numbers.Integral):
                return cls(opfn(self.value, other))
            else:
                return NotImplemented

        def rop(self, other):
            cls = type(self)
            if isinstance(other, cls):
                return cls(opfn(other.value, self.value))
            elif isinstance(other, numbers.Integral):
                return cls(opfn(other, self.value))
            else:
                return NotImplemented

        op.__name__ = f"__{opfn.__name__.strip('_')}__"
        op.__qualname__ = f"Bit.__{opfn.__name__.strip('_')}__"
        rop.__name__ = f"__r{opfn.__name__.strip('_')}__"
        rop.__qualname__ = f"Bit.__r{opfn.__name__.strip('_')}__"
        return op, rop

    @staticmethod
    def _left_operator(opfn, _operator=_operator):
        "Operator that wraps only in the left version."
        def rop(self, other):
            cls = type(self)
            if isinstance(other, cls):
                return cls(opfn(other.value, self.value))
            elif isinstance(other, numbers.Integral):
                return opfn(other, self.value)
            else:
                return NotImplemented

        rop.__name__ = f"__r{opfn.__name__.strip('_')}__"
        rop.__qualname__ = f"Bit.__r{opfn.__name__.strip('_')}__"
        return _operator(opfn)[0], rop

    @staticmethod
    def _comparator(opfn):
        "Operators that never wrap at all."
        def op(self, other):
            return opfn(self.value, other)
        op.__name__ = f"__{opfn.__name__.strip('_')}__"
        op.__qualname__ = f"Bit.__{opfn.__name__.strip('_')}__"
        return op

    @staticmethod
    def _identity(name: str):
        def op(self):
            return self
        op.__name__ = f'__{name}__'
        op.__qualname__ = f"Bit.__{name}__"
        return op

    # These operators don't all make sense for bits, but we get them all
    # anyway.

    def __hash__(self):
        return hash(self.value)

    __eq__ = _comparator(operator.eq)
    __ne__ = _comparator(operator.ne)
    __le__ = _comparator(operator.le)
    __lt__ = _comparator(operator.lt)
    __ge__ = _comparator(operator.ge)
    __gt__ = _comparator(operator.gt)

    __add__, __radd__ = _operator(operator.add)
    __sub__, __rsub__ = _operator(operator.sub)
    __mul__, __rmul__ = _operator(operator.mul)

    __truediv__, __rtruediv__ = _left_operator(operator.truediv)
    __floordiv__, __rfloordiv__ = _left_operator(operator.floordiv)
    __mod__, __rmod__ = _left_operator(operator.mod)

    def __pow__(self, other, mod=None):
        cls = type(self)
        if isinstance(other, cls):
            return cls(pow(self.value, other.value, mod))
        elif isinstance(other, numbers.Integral):
            return cls(pow(self.value, other, mod))
        else:
            return NotImplemented

    def __rpow__(self, other, mod=None):
        return pow(other, self.value, mod)

    __lshift__, __rlshift__ = _operator(operator.lshift)
    __rshift__, __rrshift__ = _operator(operator.rshift)

    __and__, __rand__ = _operator(operator.and_)
    __xor__, __rxor__ = _operator(operator.xor)
    __or__, __ror__ = _operator(operator.or_)

    def __neg__(self):
        # The other option would be identity (bitwise complement + 1 = self)
        raise TypeError("can't arithmetically negate a bit")

    __pos__ = _identity('pos')
    __abs__ = _identity('abs')

    def __invert__(self):
        """Invert the bit.

        This differs from the specification for __invert__.
        """
        return type(self)(not self.value)

    def __int__(self):
        return self.value

    __trunc__ = _identity('trunc')
    __ceil__ = _identity('ceil')
    __floor__ = _identity('floor')

    def __round__(self, ndigits=None):
        return self

    del _operator, _left_operator, _comparator, _identity
