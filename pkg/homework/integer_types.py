from __future__ import annotations

from collections.abc import Callable
import numbers
import operator
from typing import (Any, overload, TYPE_CHECKING)


if TYPE_CHECKING:
    @overload
    def _bitop(self, other: Bit) -> Bit: ...

    @overload
    def _bitop(self, other: int) -> int: ...

    def _bitop(self, other):
        return
else:
    _bitop = None


def _returning_type_of_twice[F: Callable, **A
                             ](f: F) -> Callable[[Callable[A, Any]],
                                                 Callable[A, tuple[F, F]]]:
    """Like util.returning_type_of, but returns the type twice.
    """
    return lambda fn: fn


class Bit(numbers.Integral):
    """A single bit that acts like an integer.

    Operations between bits also return bits. In operations with other types,
    bits act as integers.
    """
    # The type of value is like that because int is not a subclass of
    # numbers.Integral, in the eyes of the type checkers.
    def __init__(self, value: int | numbers.Integral):
        self._value: int
        if isinstance(value, type(self)):
            self._value = value.value
        elif isinstance(value, int):
            self._value = value % 2
        else:
            self._value = int(value) % 2

    @property
    def value(self) -> int:
        return self._value

    def __repr__(self):
        return f'{type(self).__name__}({self.value})'

    @staticmethod
    @_returning_type_of_twice(_bitop)
    def _operator(opfn):
        def op(self, other):
            cls = type(self)
            if isinstance(other, cls):
                return cls(opfn(self.value, other.value))
            elif isinstance(other, numbers.Integral):
                return opfn(self.value, other)
            else:
                return NotImplemented

        def rop(self, other):
            cls = type(self)
            if isinstance(other, cls):
                return cls(opfn(other.value, self.value))
            elif isinstance(other, numbers.Integral):
                return opfn(other, self.value)
            else:
                return NotImplemented

        op.__name__ = f"__{opfn.__name__.strip('_')}__"
        op.__qualname__ = f"Bit.__{opfn.__name__.strip('_')}__"
        rop.__name__ = f"__r{opfn.__name__.strip('_')}__"
        rop.__qualname__ = f"Bit.__r{opfn.__name__.strip('_')}__"
        return op, rop

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

    __truediv__, __rtruediv__ = _operator(operator.truediv)
    __floordiv__, __rfloordiv__ = _operator(operator.floordiv)
    __mod__, __rmod__ = _operator(operator.mod)

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

    del _operator, _comparator, _identity
