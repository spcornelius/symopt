from itertools import combinations_with_replacement

import numpy as np
import sympy as sym
from orderedset import OrderedSet
from sympy import Symbol, MatrixSymbol, sympify, DeferredVector, AtomicExpr

__all__ = []
__all__.extend([
    'as_scalars',
    'squeezed',
    'is_linear',
    'is_quadratic',
    'negated',
    'reshape_like',
    'reshape_args',
    'is_scalar',
    'fill_in_values'
])


def fill_in_values(expr, syms, vals):
    """ Replace scalar/matrix symbols with specified values in an expression.

    Parameters
    ----------
    expr : `~sympy.core.expr.Expr`
        Symbolic expression in which to find-and-replace.
    syms : `~collections.abc.Iterable` of `~typing.Union` \
            [`~sympy.core.symbol.Symbol` , \
            `~sympy.matrices.expressions.MatrixSymbol` ]
        The symbols appearing in :py:attr:`expr` to replace.
    vals :  `~collections.abc.Iterable` of `~typing.Union` \
            [`~numbers.Real`, \
             `array_like` ]
        The numeric values that should replace :py:attr:`syms`.

    Returns
    -------
    `~sympy.core.expr.Expr`
        The substituted expression.
    """
    subs = {p: float(v) if is_scalar(p) else sym.Matrix(v) for
            p, v in zip(syms, vals)}
    return expr.subs(subs)


def is_scalar(x):
    """ Test if given input is a numeric/symbolic scalar.

    Parameters
    ----------
    x : `object`
        Symbolic/numeric variable to test.

    Returns
    -------
    `bool`
        `True` if `x` is a scalar, `False` otherwise.
    """
    return isinstance(sympify(x), AtomicExpr)


def negated(fun):
    """ Negate the output of a given function

    Parameters
    ----------
    fun : `~collections.abc.Callable`
        Function with numeric or array-like output.

    Returns
    -------
    `~collections.abc.Callable`
        New function with same signature as :py:attr:`fun` that
        returns ``-fun(*args, **kwargs)``.
    """

    def wrapped(*args, **kwargs):
        return -fun(*args, **kwargs)

    return wrapped


def depends_only_on(expr, syms):
    """ Test if expr depends only on specified symbols

    Parameters
    ----------
    expr : `~sympy.core.expr.Expr`
        SymPy expression to evaluate.
    syms : `~collections.abc.Iterable` of `~typing.Union` \
            [`~sympy.core.symbol.Symbol` , \
            `~sympy.matrices.expressions.MatrixSymbol` ]
        Allowed symbols.

    Returns
    -------
    `bool`
        `True` if the free symbols in :py:attr:`expr` are contained in \
        :py:attr:`syms`, `False` otherwise.
    """
    return sympify(expr).free_symbols <= set(syms)


def squeezed(fun):
    """ Wrap an array-returning function to squeeze the output.

    Parameters
    ----------
    fun : `~collections.abc.Callable`
        Function to wrap.

    Returns
    -------
    wrapped : `~collections.abc.Callable`
        New function with same signature as :py:attr:`fun` that returns
        ``fun(*args, **kwargs).squeeze()``.
    """

    def wrapped(*args, **kwargs):
        return fun(*args, **kwargs).squeeze()

    return wrapped


def as_scalars(var):
    """ Convert a symbolic var to a list of its component scalar symbols.

    Parameters
    ----------
    var : `~typing.Union` [`~sympy.core.symbol.Symbol` \
                           `~sympy.matrices.expressions.MatrixSymbol` ]
        Symbolic variable to decompose.

    Returns
    -------
    `~typing.Union` [ `list` , `~sympy.matrices.matrices.MatrixBase` ]
        ``var.as_explicit()`` if :py:attr:`var` is an instance of \
         `~sympy.matrices.expressions.MatrixSymbol` otherwise ``[var]``.
    """
    if isinstance(var, MatrixSymbol):
        return sym.flatten(var.as_explicit())
    else:
        return [var]


def is_linear(expr, vars):
    """ Determine if expression is linear w.r.t specified symbols.

    Parameters
    ----------
    expr : `~sympy.core.expr.Expr`
        SymPy expression to test.
    vars : `~collections.abc.Iterable` of `~sympy.core.symbol.Symbol`
        The (scalar) variables of interest.

    Returns
    -------
    `bool`
        `True` if `expr` is jointly linear w.r.t. all variables in\
        :py:attr:`vars`, otherwise `False` .
    """
    pairs = combinations_with_replacement(vars, 2)
    try:
        return all(sym.Eq(sym.diff(expr, *t), 0) for t in pairs)
    except TypeError:
        return False


def is_quadratic(expr, vars):
    """ Determine if a symbolic expression is (at most) quadratic with
    respect to specified symbols.

    Parameters
    ----------
    expr : `~sympy.core.expr.Expr`
        SymPy expression to test.
    vars : `~collections.abc.Iterable` of `~sympy.core.symbol.Symbol`
        The (scalar) variables of interest.

    Returns
    -------
    `bool`
        `True` if `expr` is at most jointly quadratic w.r.t. all variables in
        :py:attr:`vars`, otherwise `False` .
    """
    vars = OrderedSet(vars)
    pairs = combinations_with_replacement(vars, 2)
    try:
        return not any((sym.diff(expr, *t).free_symbols & vars) for t in pairs)
    except TypeError:
        return False


def reshape_like(arg, s):
    """ Reshape numeric argument to match that of a symbolic variable.

    Parameters
    ----------
    arg : `~typing.Union` [ `Real`, `~numpy.ndarray` ]
        Numeric data
    s : `~typing.Union` [ `~sympy.core.symbol.Symbol`,\
                            `~sympy.matrices.expressions.MatrixSymbol` ]
        Symbolic variable whose shape :py:attr:`arg` should match.

    Returns
    -------
    `~numbers.Real` or `~numpy.ndarray`
        The reshaped numeric data.
    """
    if isinstance(s, Symbol):
        return float(arg)
    elif isinstance(s, MatrixSymbol):
        return np.asfarray(arg).reshape(s.shape)
    elif isinstance(s, DeferredVector):
        return np.asfarray(arg).flatten()
    else:
        raise TypeError(
            f"Can't understand symbolic variable {s} with type {type(s)}.")


def reshape_args(func, args):
    """ Wrap an function to pre-shape numeric param inputs correctly.

        Parameters
        ----------
        func : `~collections.abc.Callable`
            Function to wrap, with signature ``fun(x[:], *arg_vals)``
        args : `~collections.abc.Iterable` of `~typing.Union` \
                 [ `~sympy.core.symbol.Symbol`,\
                  `~sympy.matrices.expressions.MatrixSymbol` ]
            Symbolic parameters.

        Returns
        -------
        wrapped : `~collections.abc.Callable`
            New function with same signature as :py:attr:`func`, but for
            which ``argvals`` are pre-processed to match the shapes of
            the symbols in :py:attr:`args`.
    """

    def wrapped(*arg_vals):
        new_args = tuple(reshape_like(val, arg) for
                         arg, val in zip(args, arg_vals))
        return func(*new_args)

    return wrapped
