from symopt.base import SymOptExpr
import sympy as sym


class ObjectiveFunction(SymOptExpr):
    """ Symbolic (non)linear optimization objective function. """

    def __init__(self, obj, prob, **kwargs):
        """ Symbolic (non)linear optimization objective function.

        Parameters
        ----------
        obj : `~sympy.core.expr.Expr`
            Symbolic expression representing the objective function,
            in terms of :py:attr:`prob.vars` and :py:attr:`prob.params`.
        prob : `.OptimizationProblem`
            The containing optimization problem.
        **kwargs
            Keyword args to pass to `.SymOptBase`.
        """
        self.obj = sym.sympify(obj)
        super().__init__(prob, **kwargs)

    @property
    def expr(self):
        return self.obj

    @property
    def sympified(self):
        return self.obj

    def __repr__(self):
        return f"ObjectiveFunction('{self.obj}')"
