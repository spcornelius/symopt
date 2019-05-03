import sympy as sym
from itertools import chain
from sympy import MatrixSymbol

__all__ = []
__all__.extend([
    'constituent_scalars',
    'squeezed',
    'flatten_and_concat'
])


def squeezed(method):
    """ Wrap a ndarray-returning function to squeeze the output.

    Parameters
    ----------
    method : callable
        Function to wrap.

    Returns
    -------
    wrapped : callable
        Wrapped function.
    """
    def wrapped(*args, **kwargs):
        return method(*args, **kwargs).squeeze()

    return wrapped


def flatten_and_concat(*args):
    """ Concatenate a list of nested iterables into a flat list. """
    return chain.from_iterable(sym.flatten(arg) for arg in args)


def constituent_scalars(s):
    """ Convert a symbolic var to a list of its component scalar symbols.

    Parameters
    ----------
    s : Symbol or MatrixSymbol
        Symbolic variable to decompose.
    """
    if isinstance(s, MatrixSymbol):
        return s.as_explicit()
    else:
        return [s]
