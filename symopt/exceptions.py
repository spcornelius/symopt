__all__ = []
__all__.extend([
    'SymOptError',
    'IncompatibleSolverError'
])


class SymOptError(Exception):
    pass


class IncompatibleSolverError(SymOptError):
    pass
