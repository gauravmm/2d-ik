"""Type stubs for sympy module."""

from typing import Any, Callable, Sequence, TypeVar, Union, overload

# Type variables for sympy expressions
_Expr = TypeVar("_Expr", bound="Expr")

class Expr:
    """Base class for all sympy expressions."""

    def __add__(self, other: Any) -> Expr: ...
    def __radd__(self, other: Any) -> Expr: ...
    def __sub__(self, other: Any) -> Expr: ...
    def __rsub__(self, other: Any) -> Expr: ...
    def __mul__(self, other: Any) -> Expr: ...
    def __rmul__(self, other: Any) -> Expr: ...
    def __truediv__(self, other: Any) -> Expr: ...
    def __rtruediv__(self, other: Any) -> Expr: ...
    def __pow__(self, other: Any) -> Expr: ...
    def __rpow__(self, other: Any) -> Expr: ...
    def __neg__(self) -> Expr: ...
    def subs(
        self,
        substitutions: Union[
            dict[Symbol, Union[float, int, Expr]],
            list[tuple[Symbol, Union[float, int, Expr]]],
            tuple[tuple[Symbol, Union[float, int, Expr]], ...],
        ],
    ) -> Expr:
        """Substitute symbols with values in the expression."""
        ...

class Symbol(Expr):
    """A symbolic variable."""

    name: str
    def __init__(self, name: str, **assumptions: Any) -> None: ...

class Float(Expr):
    """A symbolic floating point number."""

    def __init__(self, value: Union[float, int, str] = 0) -> None: ...

class Integer(Expr):
    """A symbolic integer."""

    def __init__(self, value: int) -> None: ...

def symbols(
    names: str,
    *,
    real: bool = False,
    positive: bool = False,
    negative: bool = False,
    integer: bool = False,
    **assumptions: Any,
) -> Union[Symbol, tuple[Symbol, ...]]: ...
def cos(x: Union[Expr, float, int]) -> Expr:
    """Cosine function."""
    ...

def sin(x: Union[Expr, float, int]) -> Expr:
    """Sine function."""
    ...

def sqrt(x: Union[Expr, float, int]) -> Expr:
    """Square root function."""
    ...

def simplify(expr: Expr) -> Expr:
    """Simplify a SymPy expression."""
    ...

def diff(expr: Expr, symbol: Symbol) -> Expr:
    """Compute the derivative of an expression with respect to a symbol."""
    ...

def atan2(y: Union[Expr, float, int], x: Union[Expr, float, int]):
    """Compute the arctangent with y and x provided separately."""
    ...

def lambdify(
    args: Union[Symbol, Sequence[Symbol]],
    expr: Union[Expr, Sequence[Expr]],
    modules: Union[str, list[str], None] = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Convert a SymPy expression to a numerical function."""
    ...

class Eq:
    """Equality expression."""

    def __init__(self, lhs: Expr, rhs: Union[Expr, float, int]) -> None: ...

def nsolve(
    equations: Union[Expr, list[Expr], list[Eq]],
    symbols: Union[Symbol, list[Symbol], tuple[Symbol, ...]],
    x0: Union[list[float], dict[Symbol, float]],
    *,
    dict: bool = False,
    **kwargs: Any,
) -> Union[dict[Symbol, float], list[float], float]: ...
