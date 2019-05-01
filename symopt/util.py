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
    def wrapped(*args, **kwargs):
        return method(*args, **kwargs).squeeze()

    return wrapped


def flatten_and_concat(*args):
    return chain.from_iterable(sym.flatten(arg) for arg in args)


def constituent_scalars(s):
    # convert symbolic var to a list of its component scalar symbols
    if isinstance(s, MatrixSymbol):
        return s.as_explicit()
    else:
        return [s]
