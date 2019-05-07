from symopt.base import SymOptSymbol

__all__ = []
__all__.extend([
    'Parameter'
])


class Parameter(SymOptSymbol):
    """ Symbolic parameter. """

    def __repr__(self):
        return f"Parameter('{self.symbol}'')"
