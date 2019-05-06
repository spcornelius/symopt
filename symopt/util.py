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
    fun : callable
        Function with numeric or array-like output.

    Returns
    -------
    callable
        New function with same signature as `fun` that
        returns `-fun(*args, **kwargs)`.
    """

    def wrapped(*args, **kwargs):
        return -fun(*args, **kwargs)

    return wrapped


def depends_only_on(expr, syms):
    """ Test if expr depends only on specified symbols

    Parameters
    ----------
    expr : Expr
        SymPy expression to evaluate.
    syms : Iterable of Union(Symbol, MatrixSymbol)
        Allowed symbols.

    Returns
    -------
    bool
        True if the free symbols in `expr` are contained in `syms`,
        False otherwise.
    """
    return sympify(expr).free_symbols <= set(syms)


def squeezed(fun):
    """ Wrap an array-returning function to squeeze the output.

    Parameters
    ----------
    fun : callable
        Function to wrap.

    Returns
    -------
    wrapped : callable
        New function with same signature as `fun` that returns
        `fun(*args, **kwargs).squeeze()`.
    """
    def wrapped(*args, **kwargs):
        return fun(*args, **kwargs).squeeze()

    return wrapped


def as_scalars(var):
    """ Convert a symbolic var to a list of its component scalar symbols.

    Parameters
    ----------
    var : Union(Symbol, MatrixSymbol)
        Symbolic variable to decompose.

    Returns
    -------
    Union(list, Matrix)
        `var.as_explicit()` if `var` is a MatrixSymbol, otherwise `[var]`.
    """
    if isinstance(var, MatrixSymbol):
        return sym.flatten(var.as_explicit())
    else:
        return [var]


def chain_scalars(vars):
    """ Iterator over all scalars in a list of symbolic variables.

    Parameters
    ----------
    vars : Iterable of Union(Symbol, MatrixSymbol)
        Symbolic variables over which to iterate.

    Yields
    ------
    Symbol
        Successive constituent scalars in `vars`.
    """
    yield from chain(*map(as_scalars, vars))


def is_linear(expr, vars):
    """ Determine if expression is linear w.r.t specified symbols.

    Parameters
    ----------
    expr : Expr
        SymPy expression to test.
    vars : Iterable of Symbol
        The (scalar) variables of interest.

    Returns
    -------
    bool
        True if `expr` is jointly linear w.r.t. all variables in `vars`,
        otherwise False.
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
    expr : Expr
        SymPy expression to test.
    vars : Iterable of Symbol
        The (scalar) variables of interest.

    Returns
    -------
    bool
        True if `expr` is at most jointly quadratic w.r.t. all variables in
        `vars`, otherwise False.
    """
    vars = OrderedSet(vars)
    pairs = combinations_with_replacement(vars, 2)
    try:
        return not any((sym.diff(expr, *t).free_symbols & vars) for t in pairs)
    except TypeError:
        return False
