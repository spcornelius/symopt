from collections.abc import Sequence

from sympy import sympify, Unequality, SympifyError, Equality, LessThan, \
    StrictLessThan
from sympy.core.relational import Relational

from symopt.base import SymOptBase
from symopt.util import negated


class ConstraintCollection(Sequence):
    """ A collection of optimization constraints.

        Effectively just a list of `~.Constraint` objects. However, this
        wrapper provides some convenient functions that operate on
        an optimization problem's constraint set as a whole.
        """

    def __init__(self, cons, vars, params, wrap_using='lambdify',
                 simplify=True):
        """ A collection of optimization constraints.

        Effectively just a list of `~.Constraint` objects. However, this
        wrapper provides some convenient functions that operate on
        an optimization problem's constraint set as a whole.

        Parameters
        ----------
        cons : `~collections.abc.Iterable` of \
                `~sympy.core.relational.Relational`
            The constraints, in terms of :py:attr:`vars` and :py:attr:`params`.
        vars : `~collections.abc.Sequence` of `~typing.Union` \
                 [ `~sympy.core.symbol.Symbol`,\
                  `~sympy.matrices.expressions.MatrixSymbol` ]
            The symbolic variables.
        params : `~collections.abc.Sequence` of `~typing.Union` \
                 [ `~sympy.core.symbol.Symbol`,\
                  `~sympy.matrices.expressions.MatrixSymbol` ]
            The symbolic parameters.
        wrap_using : `str`, either 'lambdify' or 'autowrap'
            Which backend to use for wrapping the
            constraints and their derivatives for numerical
            evaluation. See :func:`~sympy.utilities.lambdify.lambdify` and
            :func:`~sympy.utilities.autowrap.autowrap` for more details.
            Defaults to 'lambdify'.
        simplify : `bool`
            If `True`, simplify constraints and their derivatives
            before wrapping as functions. Defaults to `True`.
        """
        self._cons = [Constraint(c, vars, params, wrap_using=wrap_using,
                                 simplify=simplify) for c
                      in cons]

    def __iter__(self):
        yield from self._cons

    def __contains__(self, x):
        return x in self._cons

    def __len__(self):
        return len(self._cons)

    def __getitem__(self, x):
        return self._cons[x]

    def __reversed__(self):
        yield from reversed(self._cons)

    def __str__(self):
        return str(self._cons)

    def all_linear(self):
        """ True if all Constraints in this collection are linear in their
            variables, False otherwise. """
        return all(c.is_linear() for c in self)

    def all_quadratic(self):
        """ True if all Constraints in this collection are at most quadratic
            in their variables, False otherwise. """
        return all(c.is_quadratic() for c in self)


class Constraint(SymOptBase):
    """ Symbolic (non)linear optimization constraint. """

    def __init__(self, con, vars, params, wrap_using='lambdify',
                 simplify=True):
        """ Symbolic (non)linear optimization constraint.

        Parameters
        ----------
        con : `~sympy.core.relational.Relational`
            The constraint, in terms of :py:attr:`vars` and :py:attr:`params`.
        vars : `~collections.abc.Sequence` of `~typing.Union` \
                 [ `~sympy.core.symbol.Symbol`,\
                  `~sympy.matrices.expressions.MatrixSymbol` ]
            The symbolic variables.
        params : `~collections.abc.Sequence` of `~typing.Union` \
                 [ `~sympy.core.symbol.Symbol`,\
                  `~sympy.matrices.expressions.MatrixSymbol` ]
            The symbolic parameters.
        wrap_using : `str`, either 'lambdify' or 'autowrap'
            Which backend to use for wrapping the
            constraint and its derivatives for numerical
            evaluation. See :func:`~sympy.utilities.lambdify.lambdify` and
            :func:`~sympy.utilities.autowrap.autowrap` for more details.
            Defaults to 'lambdify'.
        simplify : `bool`
            If `True`, simplify constraint and its derivatives
            before wrapping as functions. Defaults to `True`.
        """
        try:
            con = sympify(con)
        except (AttributeError, SympifyError):
            raise TypeError(f"Couldn't sympify constraint {con}.")

        if not isinstance(con, Relational):
            raise TypeError(
                f"Constraint expression {con} is not of type Relational.")
        if isinstance(con, Unequality):
            raise TypeError(
                f"Constraint expression {con} of type Unequality does not "
                f"make sense.")

        self._con = con.__class__(con.lhs - con.rhs, 0)
        super().__init__(self._con.lhs, vars, params, wrap_using=wrap_using,
                         simplify=simplify)

    def __str__(self):
        return str(self._con)

    @property
    def type(self):
        """ The type (a subclass of of `sympy.core.relational.Relational` )
            of this constraint (greater than, less than, equality, etc.)."""
        return self._con.__class__

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
