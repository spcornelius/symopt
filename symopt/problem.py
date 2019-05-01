from itertools import starmap
import numpy as np
import sympy as sym
from orderedset import OrderedSet
from sympy import sympify, Symbol, Matrix, MatrixSymbol, SympifyError, \
    GreaterThan, StrictGreaterThan, LessThan, StrictLessThan, Equality, \
    Unequality, oo
from sympy.core.relational import Relational
from sympy.utilities.autowrap import autowrap

from symopt.util import constituent_scalars, squeezed, flatten_and_concat

import scipy.optimize as scipy_opt
import ipopt as cyipopt

__all__ = []
__all__.extend([
    'OptimizationProblem'
])


class OptimizationProblem(object):

    def __init__(self, obj, vars, lb=None, ub=None,
                 constraints=None, params=None):
        self._process_vars(vars)
        self._process_params(params)
        self._process_objective(obj)
        self._process_constraints(constraints)
        self._process_bounds(lb, ub)

    @property
    def free_symbols(self):
        # Note: having params first is important for use of partial below
        return self.vars | self.params

    def depends_only_on_params(self, expr):
        # test whether expression's free symbols are only in params
        return sympify(expr).free_symbols <= self.params

    def depends_only_on_params_or_vars(self, expr):
        # test whether expression's free symbols are only in params or vars
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

    def autowrap(self, expr):
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
        if not self.depends_only_on_params_or_vars(obj):
            raise ValueError(
                "Objective function can depend only on declared vars and "
                "params.")
        self.obj = obj
        self.obj_cb = self.autowrap(self.obj)
        self.obj_grad = self._grad(self.obj)
        self.obj_grad_cb = squeezed(self.autowrap(self.obj_grad))

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
            if not self.depends_only_on_params_or_vars(c):
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
        self.con_cbs = [self.autowrap(c.lhs) for c in self.con]
        self.con_grads = [self._grad(c.lhs) for c in self.con]

        self.con_grad_cbs = [squeezed(self.autowrap(g)) for g in
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
            raise TypeError("Could not convert bounds to lists of Reals and/or"
                            "sympy expressions.")
        if len(lb) != n or len(ub) != n:
            raise ValueError(
                f"lb and ub must have the same length ({n}) as number of "
                f"scalar variables")
        for expr in lb:
            if not self.depends_only_on_params_or_vars(expr):
                raise ValueError(
                    f"Lower bound {expr} may only depend on declared params.")
        for expr in ub:
            if not self.depends_only_on_params_or_vars(expr):
                raise ValueError(
                    f"Upper bound {expr} may only depend on declared params.")
        self.lb = Matrix(lb)
        self.ub = Matrix(ub)

    def fill_in_params(self, expr, *param_vals):
        subs = {p: Matrix(v) if isinstance(p, MatrixSymbol) else v for
                p, v in zip(self.params, param_vals)}
        return expr.subs(subs)

    @squeezed
    def eval_ub(self, *param_vals):
        return np.asarray(self.fill_in_params(self.ub, *param_vals).evalf())

    @squeezed
    def eval_lb(self, *param_vals):
        return np.asarray(self.fill_in_params(self.lb, *param_vals).evalf())

    def solve(self, x0, *args, method='cyipopt', **kwargs):
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
        lb = self.eval_lb(*args)
        ub = self.eval_ub(*args)

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
