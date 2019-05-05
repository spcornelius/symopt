from itertools import starmap

import ipopt as cyipopt
import numpy as np
import scipy.optimize as scipy_opt
import sympy as sym
from orderedset import OrderedSet
from sympy import sympify, Symbol, Matrix, MatrixSymbol, SympifyError, \
    GreaterThan, StrictGreaterThan, LessThan, StrictLessThan, Equality, \
    Unequality, oo
from sympy.core.relational import Relational
from sympy.utilities.autowrap import autowrap

from symopt.util import constituent_scalars, squeezed, flatten_and_concat

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
        obj : Expr
            The objective function to optimize.
        vars : OrderedSet with Symbol or MatrixSymbol elements
            The free variables.
        params : OrderedSet with Symbol or MatrixSymbol elements
            The parameters of the objective function and/or constraints.
        cons : list with SymPy GreatherThan or Equality elements
            The constraints, converted to the form expr >= 0 or expr == 0.
        lb : list of Exprs
            The lower bounds, one for each scalar in `vars'. If
            symbolic, should depend only on `params`.
        ub : list of Exprs
            The upper bounds, one for each scalar in `vars'. If
            symbolic, should depend only on `params`.
        mode : str, one of 'max' or 'min'
            Whether the problem is a minimization or maximization problem.
        """

    def __init__(self, obj, vars, lb=None, ub=None,
                 constraints=None, params=None, mode="min"):
        """ Optimization problem with symbolic objective function, constraints,
            and/or parameters.

            Parameters
            ----------
            obj : Expr
                The objective function to optimize. Can depend on `vars` and
                also `params`.
            vars : Iterable of Symbols and/or MatrixSymbols
                The free variables.
            params : Iterable of Symbols and/or MatrixSymbols
                The parameters of the objective function and/or constraints.
            cons : Iterable of SymPy Relationals
                The constraints. Can depend on both `vars` and `params`.
            lb : Iterable of Reals and/or Exprs
                The lower bounds, one for each scalar in `vars'. If
                symbolic, should depend only on `params`.
            ub : Iterable of Reals and/or Exprs
                The upper bounds, one for each scalar in `vars'. If
                symbolic, should depend only on `params`.
            mode : str, one of 'max' or 'min'
                Whether the problem is a minimization or maximization problem.
        """
        self.mode = str(mode).lower()
        if mode not in ["min", "max"]:
            raise ValueError(
                "Optimization mode must be one of ['min', 'max'].")
        self._process_vars(vars)
        self._process_params(params)
        self._process_objective(obj)
        self._process_constraints(constraints)
        self._process_bounds(lb, ub)

    @property
    def free_symbols(self):
        """ OrderedSet: All symbols (vars + params) present in the problem."""
        # Note: having params first is important for use of partial below
        return self.vars | self.params

    def depends_only_on_params(self, expr):
        """ Test whether expression depends only on problem parameters """
        # test whether expression's free symbols are only in params
        return sympify(expr).free_symbols <= self.params

    def depends_only_on_params_and_vars(self, expr):
        """ Test whether expression depends only on problem parameters
            and/or/vars """
        return sympify(expr).free_symbols <= self.free_symbols

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
        self._scalar_vars = OrderedSet(
            flatten_and_concat(*map(constituent_scalars, vars)))
        self._dummy = MatrixSymbol('dummy', len(self._scalar_vars), 1)

    def _autowrap(self, expr):
        subs = dict(zip(self._scalar_vars, sym.flatten(self._dummy)))
        args = (self._dummy,) + tuple(self.params)
        return autowrap(expr.subs(subs), args=args)

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
        self._scalar_params = OrderedSet(
            flatten_and_concat(*map(constituent_scalars, params)))

    def _process_objective(self, obj):
        try:
            obj = sympify(obj)
        except SympifyError:
            raise TypeError("Couldn't sympify provided objective function.")
        if not self._depends_only_on_params_or_vars(obj):
            raise ValueError(
                "Objective function can depend only on declared vars and "
                "params.")
        if self.mode == "max":
            obj = -obj
        self.obj = obj
        self.obj_cb = self._autowrap(self.obj)
        self.obj_grad = self._grad(self.obj)
        self.obj_grad_cb = squeezed(self._autowrap(self.obj_grad))

    def _process_constraints(self, con):
        try:
            con = [sympify(c) for c in con]
        except (AttributeError, SympifyError):
            raise TypeError("Couldn't sympify constraint list.")
        processed = []
        for c in con:
            if not isinstance(c, Relational):
                raise TypeError(
                    f"Constraint expression {c} is not of type Relational.")
            if isinstance(c, Unequality):
                raise TypeError(
                    f"Constraint expression {c} of type Unequality does not "
                    f"make sense.")
            if not self._depends_only_on_params_or_vars(c):
                raise ValueError(
                    f"Constraint {c} may only depend on declared vars and "
                    f"params.")

            # convert to have RHS of zero, and make inequalites
            # all greater-than
            if isinstance(c, (LessThan, StrictLessThan)):
                c = GreaterThan(c.rhs - c.lhs, 0)
            elif isinstance(c, (GreaterThan, StrictGreaterThan)):
                c = GreaterThan(c.lhs - c.rhs, 0)
            else:
                c = Equality(c.lhs - c.rhs, 0)

            processed.append(c)

        self.con = Matrix(processed)
        self.con_cbs = [self._autowrap(c.lhs) for c in self.con]
        self.con_grads = [self._grad(c.lhs) for c in self.con]

        self.con_grad_cbs = [squeezed(self._autowrap(g)) for g in
                             self.con_grads]

    def _grad(self, expr):
        return Matrix([expr.diff(s) for s in self._scalar_vars])

    def _process_bounds(self, lb, ub):
        n = len(self._scalar_vars)
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
            if not self._depends_only_on_params_or_vars(expr):
                raise ValueError(
                    f"Lower bound {expr} may only depend on declared params.")
        for expr in ub:
            if not self._depends_only_on_params_or_vars(expr):
                raise ValueError(
                    f"Upper bound {expr} may only depend on declared params.")
        self.lb = lb
        self.ub = ub

    def _fill_in_params(self, expr, *param_vals):
        subs = {p: Matrix(v) if isinstance(p, MatrixSymbol) else v for
                p, v in zip(self.params, param_vals)}
        return expr.subs(subs)

    @squeezed
    def _eval_ub(self, *param_vals):
        return np.asarray(
            self._fill_in_params(Matrix(self.ub), *param_vals).evalf())

    @squeezed
    def _eval_lb(self, *param_vals):
        return np.asarray(
            self._fill_in_params(Matrix(self.lb), *param_vals).evalf())

    def solve(self, x0, *args, method='cyipopt', **kwargs):
        """ Solve optimization problem for particular parameter values.

        Parameters
        ----------
        x0 : ndarray
            The initial condition to use for the optimizer.
        args
            The parameter values to use, defined in the same order (and
            with the same shapes as in `params`). Should be Real scalars
            or Matrix objects with numerical (not symbolic) entries.
        method : str
            Which optimization backend to use. Currently supported are
            one of 'cyipopt', 'slsqp' (from scipy.optimize), and 'cobyla'
            (from scipy.optimize).
        **kwargs
            Keyword arguments to pass to the optimization backend. See the
            corresponding docs for available options.

        Returns
        -------
        sol
            Solution dictionary. See `scipy.optimize.minimize` for details.
        """
        fun, jac, bounds, constraints = self._prepare(*args)
        method = method.lower()
        if method == 'cyipopt':
            return cyipopt.minimize_ipopt(fun, x0, args=args, jac=jac,
                                          bounds=bounds,
                                          constraints=constraints, **kwargs)
        elif method == 'slsqp':
            return scipy_opt.minimize(fun, x0, args=args, jac=jac,
                                      method='SLSQP', bounds=bounds,
                                      constraints=constraints, **kwargs)
        elif method == 'cobyla':
            if any(c['type'] == 'eq' for c in constraints):
                raise ValueError(
                    "COBYLA supports only inequality constraints.")
            return scipy_opt.minimize(fun, x0, args=args, jac=jac,
                                      method='COBYLA', bounds=bounds,
                                      constraints=constraints, **kwargs)
        else:
            raise ValueError(f"Unsupported optimization method '{method}'.")

    def _prepare(self, *args):
        lb = self._eval_lb(*args)
        ub = self._eval_ub(*args)

        # replace infinities with None
        lb[lb == -np.inf] = None
        ub[ub == np.inf] = None

        bounds = list(zip(lb, ub))

        # make constraints
        def constraint_dict(c, c_fun, c_jac):
            return dict(type=('eq' if isinstance(c, Equality) else 'ineq'),
                        fun=c_fun, jac=c_jac, args=args)

        cons = zip(self.con, self.con_cbs,
                   self.con_grad_cbs)
        constraints = list(starmap(constraint_dict, cons))

        return self.obj_cb, self.obj_grad_cb, bounds, constraints
