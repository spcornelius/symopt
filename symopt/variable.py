import numpy as np
import sympy as sym
from sympy import oo, sympify

import symopt.util as util
from symopt.base import SymOptSymbol

__all__ = []
__all__.extend([
    'Variable'
])


class Variable(SymOptSymbol):
    """ Symbolic variable in an `.OptimizationProblem`.

        Wraps a `~sympy.core.symbols.Symbol` or
        `~sympy.matrices.expressions.MatrixSymbol` to include bound
        information, etc.

        Attributes
        ----------
        lb: `~sympy.core.expr.Expr` or \
            `~sympy.matrices.immutable.ImmutableDenseMatrix`
            The variable's lower bound(s).
        ub: `~sympy.core.expr.Expr` or \
            `~sympy.matrices.immutable.ImmutableDenseMatrix`
            The variable's upper bound(s).
        """

    def __init__(self, symbol, prob, lb=None, ub=None):
        """ Symbolic variable in an `.OptimizationProblem`.

        Parameters
        ----------
        symbol: `~sympy.core.symbols.Symbol` or \
                    `~sympy.matrices.expressions.MatrixSymbol`
            The symbol representing the variable.
        prob: `.OptimizationProblem`
            The containing optimization problem.
        lb: `~numbers.Real`, `~sympy.core.expr.Expr`,\
                or `~collections.abc.Iterable`
            The lower bound(s). If symbolic, should depend only on
            :py:attr:`prob.params`. If :py:attr:`symbol` is
            non-scalar, :py:attr:`lb` should be coercible to a
            `~sympy.matrices.immutable.ImmutableDenseMatrix` with the same
            shape. Defaults to `None` (unbounded below).
        ub: `~numbers.Real`, `~sympy.core.expr.Expr`,\
                or `~collections.abc.Iterable`
            The upper bound(s). If symbolic, should depend only on
            :py:attr:`prob.params`. If :py:attr:`symbol` is
            non-scalar, :py:attr:`lb` should be coercible to a
            `~sympy.matrices.immutable.ImmutableDenseMatrix` with the same
            shape.D efaults to `None` (unbounded above).
        """
        super().__init__(symbol, prob)
        self.lb = lb
        self.ub = ub

    def _process_bound(self, b):
        if self.is_scalar:
            if not util.is_scalar(b):
                raise TypeError(
                    f"Variable {self.symbol} is a scalar but non-scalar bounds"
                    f"were provided.")
        else:
            shape = self.symbol.shape
            try:
                b = sym.ImmutableMatrix(b).reshape(*shape)
            except (TypeError, ValueError):
                b = b * sym.ones(*shape)

        b = sympify(b)
        if not self.prob.depends_only_on_params(b):
            raise ValueError(
                "Bounds can depend only on the symbols in prob.params")
        return b

    @property
    def lb(self):
        """ `~sympy.core.expr.Expr` or
            `~sympy.matrices.immutable.ImmutableDenseMatrix`: the lower bound
            for this variable. """
        return self._lb

    @property
    def ub(self):
        """ `~sympy.core.expr.Expr` or
            `~sympy.matrices.immutable.ImmutableDenseMatrix`: the upper bound
            for this variable. """
        return self._ub

    @lb.setter
    def lb(self, new_lb):
        if new_lb is None:
            self._lb = -oo
        else:
            self._lb = self._process_bound(new_lb)

    @ub.setter
    def ub(self, new_ub):
        if new_ub is None:
            self._ub = oo
        else:
            self._ub = self._process_bound(new_ub)

    @property
    def is_bounded(self):
        """ `bool`: `True` if the `~.Variable` has finite bounds in either
                    direction. """
        lb = np.atleast_1d(self.lb)
        ub = np.atleast_1d(self.ub)
        return np.any(lb != -oo) or np.any(ub != oo)

    def eval_bounds(self, *param_vals):
        """ Evaluate variable bounds at specified parameter values.

        Parameters
        ----------
        *param_vals
            Params at which to numerically evaluate problem bounds. Should
            be provided in same the same order as :py:attr:`self.params`.

        Returns
        -------
        `tuple` of `~numbers.Real` or \
            `~sympy.matrices.immutable.ImmutableDenseMatrix`
            The numeric lower, upper bounds for this variable evaluated
            at :py:attr:`param_vals`.
        """
        lb = util.fill_in_values(self.lb, self.prob.params, param_vals)
        ub = util.fill_in_values(self.ub, self.prob.params, param_vals)
        return lb, ub

    def __repr__(self):
        if self.is_scalar:
            return f"Variable('{self.symbol}')"
        else:
            nrow, ncol = self.symbol.shape
            return f"Variable('{self.symbol}', {nrow}, {ncol})"
