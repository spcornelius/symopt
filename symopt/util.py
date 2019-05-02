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
    """ return x.squeeze() from a function that returns a numpy
        array x """
    def wrapped(*args, **kwargs):
        return method(*args, **kwargs).squeeze()

    return wrapped


def flatten_and_concat(*args):
    """ concatenate a list of nested lists into a flat list """
    return chain.from_iterable(sym.flatten(arg) for arg in args)


def constituent_scalars(s):
    """" convert symbolic var s to a list of its component scalar symbols """
    if isinstance(s, MatrixSymbol):
        return s.as_explicit()
    else:
        return [s]
