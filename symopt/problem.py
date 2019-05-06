import numpy as np
from orderedset import OrderedSet
from sympy import sympify, Symbol, Matrix, MatrixSymbol, SympifyError, oo

from symopt.constraint import ConstraintCollection
from symopt.objective import ObjectiveFunction
from symopt.solvers import solve
from symopt.util import chain_scalars, depends_only_on

__all__ = []
__all__.extend([
    'OptimizationProblem'
])


class OptimizationProblem(object):
    """ Optimization problem with symbolic objective function, constraints,
        and/or parameters.

        The associated evaluation functions are created automatically,
        including those of the relevant derivatives (objective function
        gradient, constraint gradients, etc.), which are automatically
        derived from the symbolic expressions.

        Attributes
        ----------
        obj : `.ObjectiveFunction`
            The objective function to optimize.
        vars : `~orderedset.OrderedSet` of `~typing.Union` \
                [`~sympy.core.symbol.Symbol`,\
                 `~sympy.matrices.expressions.MatrixSymbol` ]
            The free variables.
        params : `~orderedset.OrderedSet` of `~typing.Union` \
                [`~sympy.core.symbol.Symbol`,\
                 `~sympy.matrices.expressions.MatrixSymbol` ]
            The parameters of the objective function and/or constraints.
        cons : `.ConstraintCollection`
            The `.Constraint` objects representing the constraints of
            the problem, after being converted to the form ``expr >= 0``
            (inequalities) or ``expr == 0`` (equality).
        lb : `list` of `~sympy.core.expr.Expr`
            The lower bounds, one for each scalar in `vars`. If
            symbolic, should depend only on `params`.
        ub : `list` of `~sympy.core.expr.Expr`
            The upper bounds, one for each scalar in `vars`. If
            symbolic, should depend only on `params`.
        mode : `str`, either 'max' or 'min'
            Whether the problem is a minimization or maximization problem.
        """

    def __init__(self, obj, vars, lb=None, ub=None,
                 cons=None, params=None, mode="min"):
        """ Optimization problem with symbolic objective function, constraints,
            and/or parameters.

            Parameters
            ----------
            obj : `~sympy.core.expr.Expr`
                The objective function to optimize. Can depend on `vars` and
                also `params`.
            vars : `~collections.abc.Iterable` of `~typing.Union` \
                    [`~sympy.core.symbol.Symbol`,\
                     `~sympy.matrices.expressions.MatrixSymbol` ]
                The free variables.
            params : `~collections.abc.Iterable` of `~typing.Union` \
                    [`~sympy.core.symbol.Symbol`,\
                     `~sympy.matrices.expressions.MatrixSymbol` ]
                The parameters of the objective function and/or constraints.
            cons : `list` of :class:`~sympy.core.relational.Relational`
                The constraints. Can depend on both `vars` and `params`.
            lb : `~collections.abc.Iterable` of `~typing.Union` \
                [ `~numbers.Real` , `~sympy.core.expr.Expr` ]
                The lower bounds, one for each scalar in `vars`. If
                symbolic, should depend only on `params`.
            ub : `~collections.abc.Iterable` of `~typing.Union` \
                [ `~numbers.Real` , `~sympy.core.expr.Expr` ]
                The upper bounds, one for each scalar in `vars`. If
                symbolic, should depend only on `params`.
            mode : `str`, either 'max' or 'min'
                Whether the objective function should be minimized or
                maximized (default 'min').
        """
        self.mode = str(mode).lower()
        if mode not in ["min", "max"]:
            raise ValueError(
                "Optimization mode must be one of ['min', 'max'].")
        self._process_vars(vars)
        self._process_params(params)
        self._process_bounds(lb, ub)
        self.obj = ObjectiveFunction(obj, self.vars, self.params)
        self.cons = ConstraintCollection(cons, self.vars, self.params)

    def _process_vars(self, vars):
        if vars is None:
            vars = []
        try:
            vars = [sympify(v) for v in vars]
        except (AttributeError, SympifyError):
            raise TypeError("Couldn't sympify variable list.")
        for v in vars:
            if not isinstance(v, (Symbol, MatrixSymbol)):
                raise TypeError(
                    f"Variable {v} is not a Symbol nor MatrixSymbol.")
        self.vars = OrderedSet(vars)

    def _process_params(self, params):
        if params is None:
            params = []
        try:
            params = [sympify(p) for p in params]
        except (AttributeError, SympifyError):
            raise TypeError("Couldn't sympify parameter list.")
        for p in params:
            if not isinstance(p, (Symbol, MatrixSymbol)):
                raise TypeError(
                    f"Parameter {p} is not a Symbol nor MatrixSymbol.")
        self.params = OrderedSet(params)

    def _process_bounds(self, lb, ub):
        scalar_vars = list(chain_scalars(self.vars))
        n = len(scalar_vars)
        if lb is None:
            lb = [-oo for _ in range(n)]
        if ub is None:
            ub = [oo for _ in range(n)]
        try:
            lb = [sympify(expr) for expr in lb]
            ub = [sympify(expr) for expr in ub]
        except (AttributeError, TypeError, SympifyError):
            raise TypeError("Could not convert bounds to a lists of "
                            "sympy expressions.")
        if len(lb) != n or len(ub) != n:
            raise ValueError(
                f"Bounds must have the same length ({n}) as number of "
                f"scalar variables.")

        for expr in lb:
            if not depends_only_on(expr, self.params):
                raise ValueError(
                    f"Lower bound {expr} may only depend on declared params.")
        for expr in ub:
            if not depends_only_on(expr, self.params):
                raise ValueError(
                    f"Upper bound {expr} may only depend on declared params.")
        self.lb = np.array(lb, dtype='object')
        self.ub = np.array(ub, dtype='object')

    def _fill_in_params(self, expr, *param_vals):
        subs = {p: Matrix(v) if isinstance(p, MatrixSymbol) else v for
                p, v in zip(self.params, param_vals)}
        return expr.subs(subs)

    def eval_bounds(self, *param_vals):
        """ Evaluate parametric bounds at specified parameter values.

        Parameters
        ----------
        *param_vals
            Params at which to numerically evaluate problem bounds. Should
            be provided in same the same order as `self.params`.

        Returns
        -------
        `tuple`
            Two `numpy.ndarray` s corresponding to the lower/upper
            bounds for each scalar variable in the problem, in sequence."""
        lb = np.asarray(
            self._fill_in_params(Matrix(self.lb), *param_vals).evalf())
        ub = np.asarray(
            self._fill_in_params(Matrix(self.ub), *param_vals).evalf())
        return lb.squeeze(), ub.squeeze()

    def solve(self, x0, *args, method='cyipopt', **kwargs):
        """ Solve optimization problem for particular parameter values.

        Parameters
        ----------
        x0 : `numpy.ndarray`
            The initial guess for the optimizer.
        *args
            The parameter values to use, defined in the same order (and
            with the same shapes as in `params`). Should be `~numbers.Real`
            scalars or `~sympy.matrices.matrices.MatrixBase` objects with
            `~numbers.Real` entries.
        method : `str`
            Which optimization backend to use. Currently supported are
            one of 'ipopt', 'slsqp' (from :mod:`scipy.optimize`), and
            'cobyla' (from :mod:`scipy.optimize`).
        **kwargs
            Keyword arguments to pass to the optimization backend. See the
            corresponding docs for available options.

        Returns
        -------
        sol
            Solution dictionary. See :func:`scipy.optimize.minimize`
            for details.
        """
        try:
            return solve[method](self, x0, *args, **kwargs)
        except KeyError:
            raise ValueError(f"Unknown optimization method '{method}'.")
