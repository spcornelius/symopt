from scipy.optimize import minimize
from sympy import Equality

from symopt.util import negated

__all__ = []
__all__.extend([
    'prepare_scipy',
    'solve_slsqp',
    'solve_cobyla'
])


def prepare_scipy(prob, *param_vals):
    """ Convert an `.OptimizationProblem` to inputs for\
        :func:`scipy.optimize.minimize`.

    Parameters
    ----------
    prob : `.OptimizationProblem`
        Problem to solve
    *param_vals
        Numerical values for problem parameters, supplied with the same order
        and types as :py:attr:`prob.vars`.

    Returns
    -------
    (`~collections.abc.Callable`, `~collections.abc.Callable`, \
     `list` of `dict` s, `list` of `tuple` s)
        Objective function callback, jacobian callback, constraint
        dictionaries, and the lower/upper bound for each variable. See
        documentation for :func:`scipy.optimize.minimize` for more details."""
    fun = prob.obj.cb
    jac = prob.obj.grad_cb

    if prob.mode == 'max':
        fun = negated(fun)
        jac = negated(jac)

    cons = [c.as_scipy_dict(*param_vals) for c in prob.cons]
    lb, ub = prob.eval_bounds(*param_vals)
    return fun, jac, cons, list(zip(lb, ub))


def solve_slsqp(prob, x0, *param_vals, **kwargs):
    """ Solve an optimization problem using SciPy's SLSQP method. """

    fun, jac, cons, bounds = prepare_scipy(prob, *param_vals)
    return minimize(fun, x0, args=param_vals, method='SLSQP', jac=jac,
                    bounds=bounds, constraints=cons, **kwargs)


def solve_cobyla(prob, x0, *param_vals, **kwargs):
    """ Solve an optimization problem using SciPy's COBYLA method. """

    if any(v.is_bounded for v in prob.vars):
        raise ValueError("COBYLA does not support variable lb/ub. Recast as "
                         "constraints.")
    if any(c.type is Equality for c in prob.cons):
        raise ValueError("COBYLA supports only inequality constraints")

    # COBYLA doesn't use jac or bounds
    fun, _, cons, _ = prepare_scipy(prob, *param_vals)
    return minimize(fun, x0, args=param_vals, method='COBYLA',
                    constraints=cons, **kwargs)
