from itertools import chain

import numpy as np
from orderedset import OrderedSet
from sympy import sympify

from symopt.constraint import Constraint
from symopt.objective import ObjectiveFunction
from symopt.parameter import Parameter
from symopt.solvers import solve
from symopt.variable import Variable

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
        """

    def __init__(self, mode='min', wrap_using='lambdify', simplify=True):
        """ Optimization problem with symbolic objective function, constraints,
            and/or parameters.

            Parameters
            ----------
            mode : `str`, either 'max' or 'min'
                Whether the objective function should be minimized or
                maximized (default 'min').
            wrap_using : `str`, either 'lambdify' or 'autowrap'
                Which backend to use for wrapping the
                objective/constraint/derivative expressions for numerical
                evaluation. See :func:`~sympy.utilities.lambdify.lambdify` and
                :func:`~sympy.utilities.autowrap.autowrap` for more details.
                Defaults to 'lambdify'.
            simplify : `bool`
                If `True`, simplify all symbolic expressions before wrapping
                as functions. Defaults to `True`.
        """
        self.mode = mode
        self._obj = None
        self.simplify = simplify
        self.wrap_using = wrap_using
        self._vars = OrderedSet()
        self._params = OrderedSet()
        self._cons = OrderedSet()

    @property
    def mode(self):
        """ `str` : either 'max' or 'min', depending on whether the problem is
                    a minimization or maximization problem. """
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        new_mode = str(new_mode).lower()
        if new_mode not in ['min', 'max']:
            raise ValueError(f"{new_mode} is not a valid optimization mode.")
        self._mode = new_mode

    @property
    def obj(self):
        """ `.ObjectiveFunction` : Problem objective function. """
        return self._obj

    @obj.setter
    def obj(self, new_obj):
        if new_obj is not None:
            new_obj = ObjectiveFunction(new_obj, self,
                                        wrap_using=self.wrap_using,
                                        simplify=self.simplify)
        self._obj = new_obj

    @property
    def vars(self):
        """ `~orderedset.OrderedSet` of `.Variable` : Problem parameters. """
        return self._vars

    @property
    def var_scalars(self):
        """ `list` of `~sympy.core.symbol.Symbol` :
             Constituent scalars of all variables in problem. """
        return list(chain.from_iterable(var.as_scalars() for var in self.vars))

    @property
    def params(self):
        """ `~orderedset.OrderedSet` of `.Parameter` :
            Problem parameters. """
        return self._params

    @property
    def cons(self):
        """ `~orderedset.OrderedSet` of `.Constraint` :
            Problem constraints. """
        return self._cons

    def add_variable(self, var, lb=None, ub=None):
        """ Add a symbolic variable to the problem.

        Parameters
        ----------
        var : `~sympy.core.symbol.Symbol` or \
              `~sympy.matrices.expressions.MatrixSymbol`
            Symbolic variable to add.
        lb : `~numbers.Real` , `~sympy.core.expr.Expr` , \
                or `~collections.abc.Iterable`
            The lower bound(s). If symbolic, should depend only on
            :py:attr:`self.params`. If :py:attr:`symbol` is
            non-scalar, :py:attr:`lb` should be coercible to a
            `~sympy.matrices.immutable.ImmutableDenseMatrix` with the same
            shape. Defaults to `None` (unbounded below).
        ub : `~numbers.Real` , `~sympy.core.expr.Expr` , \
                or `~collections.abc.Iterable`
            The upper bound(s). If symbolic, should depend only on
            :py:attr:`self.params`. If :py:attr:`symbol` is
            non-scalar, :py:attr:`lb` should be coercible to a
            `~sympy.matrices.immutable.ImmutableDenseMatrix` with the same
            shape. Defaults to `None` (unbounded above).
        """
        self._vars.add(Variable(var, self, lb=lb, ub=ub))

    def add_variables_from(self, var_bunch, lb=None, ub=None):
        """ Add multiple new symbolic variables to the problem.

        Parameters
        ----------
        var_bunch : `~collections.abc.Iterable of \
                    `~sympy.core.symbol.Symbol` or \
                    `~sympy.matrices.expressions.MatrixSymbol`
            Symbolic variables to add.
        lb : `~numbers.Real` , `~sympy.core.expr.Expr` , \
                or `~collections.abc.Iterable`
            The lower bound(s). If symbolic, should depend only on
            :py:attr:`self.params`. If :py:attr:`symbol` is
            non-scalar, :py:attr:`lb` should be coercible to a
            `~sympy.matrices.immutable.ImmutableDenseMatrix` with the same
            shape. The value for :py:attr:`lb` is assumed to apply to all
            new variables in :py:attr:`var_bunch`. Defaults to `None` (
            unbounded below).
        ub : `~numbers.Real` , `~sympy.core.expr.Expr` , \
                or `~collections.abc.Iterable`
            The upper bound(s). If symbolic, should depend only on
            :py:attr:`self.params`. If :py:attr:`symbol` is
            non-scalar, :py:attr:`lb` should be coercible to a
            `~sympy.matrices.immutable.ImmutableDenseMatrix` with the same
            shape. he value for :py:attr:`ub` is assumed to apply to all
            new variables in :py:attr:`var_bunch`.
            Defaults to `None` (unbounded above).
        """
        for var in var_bunch:
            self.add_variable(var, lb=lb, ub=ub)

    def add_constraint(self, con):
        """ Add a new symbolic constraint to the problem.

        Parameters
        ----------
        con : `~sympy.core.relational.Relational`
            The new constraint, as a :mod:`sympy` (in)equalities in terms of
            :py:attr:`prob.vars` and :py:attr:`prob.params`. Note:
            `~sympy.core.relational.Eq` (not ``==``) should be used to define
            equality constraints.
        """
        self._cons.add(Constraint(con, self, wrap_using=self.wrap_using,
                                  simplify=self.simplify))

    def add_constraints_from(self, con_bunch):
        """ Add multiple new symbolic constraints to the problem.

        Parameters
        ----------
        con_bunch : `~collections.abc.Iterable` of \
                    `~sympy.core.relational.Relational`
            The new constraints, as :mod:`sympy` (in)equalities in terms of
            :py:attr:`prob.vars` and :py:attr:`prob.params`. Note:
            `~sympy.core.relational.Eq` (not ``==``) should be used to define
            equality constraints.
        """
        for con in con_bunch:
            self.add_constraint(con)

    def add_parameter(self, param):
        """ Add a new symbolic parameter to the problem.

        Parameters
        ----------
        param : `~sympy.core.symbol.Symbol` or \
                `~sympy.matrices.expressions.MatrixSymbol`
            Symbolic parameter to add.
        """
        self._params.add(Parameter(param, self))

    def add_parameters_from(self, param_bunch):
        """ Add multiple new symbolic parameter to the problem.

        Parameters
        ----------
        param_bunch : `~collections.abc.Iterable` of \
                      `~sympy.core.symbol.Symbol` or \
                      `~sympy.matrices.expressions.MatrixSymbol`
            Symbolic parameter to add.
        """
        for param in param_bunch:
            self.add_parameter(param)

    def eval_bounds(self, *param_vals):
        """ Evaluate lower/upper variable bounds the problem at specified
            parameter values as flattened arrays.

        Parameters
        ----------
        *param_vals
            Numeric values for :py:attr:`self.params`. Should be either
            `~numbers.Real`, or `array_like` depending on whether the
            parameter is scalar or not.

        Returns
        -------
        (`~numpy.ndarray`, `~numpy.ndarray`)
            Arrays corresponding to the lower/upper bounds for each scalar
            variable in :py:attr:`self.var_scalars`.
        """
        bounds = [v.eval_bounds(*param_vals) for v in self.vars]
        lb, ub = tuple(zip(*bounds))
        lb = np.hstack(
            tuple(np.atleast_1d(_lb).flatten().astype('float') for _lb in lb))
        ub = np.hstack(
            tuple(np.atleast_1d(_ub).flatten().astype('float') for _ub in ub))
        return lb, ub

    def depends_only_on_params_or_vars(self, expr):
        """ Test if :py:attr:`expr` depends only problem variables
            and/or parameters

        Parameters
        ----------
        expr : `~sympy.core.expr.Expr`
            SymPy expression to evaluate.

        Returns
        -------
        `bool`
            `True` if the free symbols in :py:attr:`expr` are contained in \
            :py:attr:`self.params` and :py:attr:`self.vars`, `False` otherwise.
        """
        return sympify(expr).free_symbols <= self.params | self.vars

    def depends_only_on_params(self, expr):
        """ Test if :py:attr:`expr` depends only problem parameters

        Parameters
        ----------
        expr : `~sympy.core.expr.Expr`
            SymPy expression to evaluate.

        Returns
        -------
        `bool`
            `True` if the free symbols in :py:attr:`expr` are contained in \
            :py:attr:`self.params`, `False` otherwise.
        """
        return sympify(expr).free_symbols <= self.params

    def solve(self, x0, *param_vals, method='cyipopt', **kwargs):
        """ Solve optimization problem for particular parameter values.

        Parameters
        ----------
        x0 : `array_like`
            The initial guess for the optimizer.
        *param_vals
            The parameter values to use, defined in the same order (and
            with the same shapes as in :py:attr:`self.params`). Should be
            `~numbers.Real` scalars or `~sympy.matrices.matrices.MatrixBase`
            objects with `~numbers.Real` entries.
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
        if not vars or self.obj is None:
            raise ValueError(
                "Can't solve OptimizationProblem without variables and an "
                "objective function.")
        try:
            return solve[method](self, x0, *param_vals, **kwargs)
        except KeyError:
            raise ValueError(f"Unknown optimization method '{method}'.")
