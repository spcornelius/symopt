import numpy as np
from scipy.optimize import minimize
from sympy import Equality, oo

from symopt.util import negated

__all__ = []
__all__.extend([
    'prepare_scipy',
    'solve_slsqp',
    'solve_cobyla'
])


def prepare_scipy(prob, *args):
    """ Convert an `.OptimizationProblem` to inputs for\
        :func:`scipy.optimize.minimize`.

    Parameters
    ----------
    prob : `.OptimizationProblem`
        Problem to solve
    *args
        Numerical values for problem parameters, supplied with the same order
        and types as ``prob.vars``.

    Returns
    -------
    `tuple`
        Objective function callback, constraint dictionaries, jacobian
        callback and lower/upper bounds for each variable. See
        documentation for :func:`scipy.optimize.minimize` for more details."""
    fun = prob.obj.cb
    jac = prob.obj.grad_cb

    if prob.mode == 'max':
        fun = negated(fun)
        jac = negated(jac)

    cons = [c.as_scipy_dict(*args) for c in prob.cons]
    lb, ub = prob.eval_bounds(*args)
    return fun, cons, jac, list(zip(lb, ub))


def solve_slsqp(prob, x0, *args, **kwargs):
    """ Solve an optimization problem using SciPy's SLSQP method. """
    return _solve_scipy(prob, x0, *args, method='SLSQP', **kwargs)


def solve_cobyla(prob, x0, *args, **kwargs):
    """ Solve an optimization problem using SciPy's COBYLA method. """
    if not all(np.logical_and(prob.ub == oo, prob.lb == -oo)):
        raise ValueError("COBYLA does not support variable lb/ub. Recast as "
                         "constraints.")
    if any(c.type is Equality for c in prob.cons):
        raise ValueError("COBYLA supports only inequality constraints")
    return _solve_scipy(prob, x0, *args, method='COBYLA', **kwargs)


def _solve_scipy(prob, x0, *args, method='SLSQP', **kwargs):
    fun, cons, jac, bounds = prepare_scipy(prob, *args)
    return minimize(fun, x0, args=args, jac=jac,
                    method=method, bounds=bounds,
                    constraints=cons, **kwargs)
