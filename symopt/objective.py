from symopt.base import SymOptBase


class ObjectiveFunction(SymOptBase):
    """ Symbolic (non)linear optimization objective function. """

    def __init__(self, expr, vars, params, wrap_using='lambdify'):
        """ Symbolic (non)linear optimization objective function.

        Parameters
        ----------
        con : `~sympy.core.expr.Expr`
            The objective function, in terms of :py:attr:`vars` and
            :py:attr:`params`.
        vars : `~collections.abc.Sequence` of `~typing.Union` \
                 [ `~sympy.core.symbol.Symbol`,\
                  `~sympy.matrices.expressions.MatrixSymbol` ]
            The symbolic variables.
        params : `~collections.abc.Sequence` of `~typing.Union` \
                 [ `~sympy.core.symbol.Symbol`,\
                  `~sympy.matrices.expressions.MatrixSymbol` ]
            The symbolic parameters.
        wrap_using : `str`, either 'lambdify' or 'autowrap'
                Which backend to use for wrapping the objective function
                and its derivatives for numerical evaluation.
                See :func:`~sympy.utilities.lambdify.lambdify` and
                :func:`~sympy.utilities.autowrap.autowrap` for more details.
                Defaults to 'lambdify'.
        """
        super().__init__(expr, vars, params, wrap_using=wrap_using)

    def __str__(self):
        return str(self._con)
