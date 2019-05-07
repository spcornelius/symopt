import numpy as np
from scipy.optimize import OptimizeResult
from sympy import GreaterThan, StrictGreaterThan, LessThan, StrictLessThan
import symopt.config as config

__all__ = []
__all__.extend([
    'solve_ipopt'
])

# IPopt regards any x with |x| >= 10^19 to be infinite
INF = 10.0 ** 19


# noinspection PyPackageRequirements
def solve_ipopt(prob, x0, *param_vals, **kwargs):
    """ Solve a given `.OptimizationProblem` using Ipopt. """
    if config.HAS_IPOPT:
        from ipopt import problem
    else:
        raise ModuleNotFoundError(
            "Can't solve problems with the ipopt backend; optional dependency "
            "'ipopt' was not found.")

    if 'print_level' not in kwargs:
        kwargs['print_level'] = 0
    kwargs['nlp_scaling_method'] = 'user-scaling'
    n = len(x0)
    m = len(prob.cons)

    # (scalarized) variable bounds
    lb, ub = prob.eval_bounds(*param_vals)

    # constraint bounds
    cl = [-INF if c.type in (LessThan, StrictLessThan) else 0 for
          c in prob.cons]
    cu = [INF if c.type in (GreaterThan, StrictGreaterThan) else 0 for c in
          prob.cons]
    idx = np.triu_indices(n)

    class IpoptProblem(problem):

        def __init__(self):
            super().__init__(n, m, lb=lb, ub=ub, cl=cl, cu=cu)
            self.nfev = 0
            self.njev = 0
            self.nit = 0

        def objective(self, y):
            self.nfev += 1
            return prob.obj.cb(y, *param_vals)

        @staticmethod
        def constraints(y):
            return [c.cb(y, *param_vals) for c in prob.cons]

        def gradient(self, y):
            self.njev += 1
            return prob.obj.grad_cb(y, *param_vals)

        @staticmethod
        def jacobian(y):
            return np.hstack([c.grad_cb(y, *param_vals) for c in prob.cons])

        @staticmethod
        def hessian(y, lam, obj):
            res = obj * prob.obj.hess_cb(y, *param_vals)
            for L, c in zip(lam, prob.cons):
                res += L * c.hess_cb(y, *param_vals)
            return res[idx]

        @staticmethod
        def hessianstructure():
            return idx

        # noinspection PyUnusedLocal
        def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                         mu, d_norm, regularization_size, alpha_du, alpha_pr,
                         ls_trials):
            self.nit = iter_count

    ipopt_prob = IpoptProblem()
    if prob.mode == 'max':
        ipopt_prob.setProblemScaling(obj_scaling=-1.0)
    for k, v in kwargs.items():
        ipopt_prob.addOption(k, v)

    x, info = ipopt_prob.solve(x0)
    return OptimizeResult(x=x, success=info['status'] == 0,
                          status=info['status'],
                          message=info['status_msg'],
                          fun=info['obj_val'],
                          info=info,
                          nfev=ipopt_prob.nfev,
                          njev=ipopt_prob.njev,
                          nit=ipopt_prob.nit)
