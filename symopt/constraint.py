from sympy import sympify, Unequality, SympifyError, Equality, LessThan, \
    StrictLessThan
from sympy.core.relational import Relational

from symopt.base import SymOptExpr
from symopt.util import negated

__all__ = []
__all__.extend([
    'Constraint'
])


class Constraint(SymOptExpr):
    """ Symbolic (non)linear optimization constraint. """

    def __init__(self, rel, prob, **kwargs):
        """ Symbolic (non)linear optimization constraint.

        Parameters
        ----------
        rel : `~sympy.core.relational.Relational`
            The constraint in terms of :py:attr:`prob.vars` and
            :py:attr:`prob.params`. Note:  `~sympy.core.relational.Eq`
            (not ``==``) should be used to define equality constraints.
        prob : `.OptimizationProblem`
            The containing optimization problem.
        **kwargs
            Keyword args to pass to `.SymOptBase`.
        """

        try:
            rel = sympify(rel)
        except (AttributeError, SympifyError):
            raise TypeError(f"Couldn't sympify constraint {rel}.")

        if not isinstance(rel, Relational):
            raise TypeError(
                f"Constraint expression {rel} is not of type Relational.")
        if isinstance(rel, Unequality):
            raise TypeError(
                f"Constraint expression {rel} of type Unequality does not "
                f"make sense.")

        self.rel = rel.__class__(rel.lhs - rel.rhs, 0)
        super().__init__(prob, **kwargs)

    def __repr__(self):
        return f"Constraint('{self.rel}')"

    @property
    def expr(self):
        return self.rel.lhs

    @property
    def sympified(self):
        return self.rel

    @property
    def lhs(self):
        """ The left hand side of the constraint (should always be 0). """
        return self.rel.lhs

    @property
    def rhs(self):
        """ The right hand side of the constraint (should always be 0). """
        return self.rel.rhs

    @property
    def type(self):
        """ The type (a subclass of of `sympy.core.relational.Relational` )
            of this constraint (greater than, less than, equality, etc.)."""
        return self.rel.__class__

    def as_scipy_dict(self, *args):
        """ Represent the constraint as a dictionary for use with \
            :func:`scipy.optimize.minimize`.

        Parameters
        ----------
        *args
            Parameter values.

        Returns
        -------
        `dict`
            Constraint dictionary. See documentation for\
            :func:`scipy.optimize.minimize` for details.
        """
        fun = self.cb
        jac = self.grad_cb
        if self.type in [LessThan, StrictLessThan]:
            fun = negated(fun)
            jac = negated(jac)
        return dict(type=('eq' if self.type is Equality else 'ineq'),
                    fun=fun, jac=jac, args=args)
