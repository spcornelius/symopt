from symopt.base import SymOptBase


class ObjectiveFunction(SymOptBase):
    """ Symbolic (non)linear optimization objective function. """

    def __init__(self, expr, vars, params):
        super().__init__(expr, vars, params)

    def __str__(self):
        return str(self._con)
