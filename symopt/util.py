import sympy as sym
from itertools import chain
from orderedset import OrderedSet
from sympy import MatrixSymbol, sympify
from itertools import combinations_with_replacement


__all__ = []
__all__.extend([
    'as_scalars',
    'chain_scalars',
    'squeezed',
    'is_linear',
    'is_quadratic',
    'depends_only_on',
    'negated'
])


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


def chain_scalars(vars):
    """ Iterator over all scalars in a list of symbolic variables.

    Parameters
    ----------
    vars : `~collections.abc.Iterable` of `~typing.Union` \
            [`~sympy.core.symbol.Symbol` \
             `~sympy.matrices.expressions.MatrixSymbol` ]
        Symbolic variables over which to iterate.

    Yields
    ------
    `~sympy.core.symbol.Symbol`
        Successive constituent scalars in :py:attr:`vars`.
    """
    yield from chain(*map(as_scalars, vars))


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
