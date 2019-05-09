from itertools import product

import numpy as np
import pytest
import sympy as sym

import symopt.config as config
from symopt.problem import OptimizationProblem

tol = 1.0e-8

wrap_using_values = ['lambdify', 'autowrap']


def needs_ipopt(test_func):
    def new_test_func(solver, wrap_using):
        if solver == 'ipopt' and not config.HAS_IPOPT:
            pytest.skip(
                "Test requires optional dependency ipopt, which is not "
                "installed.")
        else:
            return test_func(solver, wrap_using)

    return new_test_func


@pytest.mark.parametrize("solver,wrap_using",
                         product(["ipopt", "slsqp"], wrap_using_values))
@needs_ipopt
def test_prob18(solver, wrap_using):
    """ problem 18 from the Hock-Schittkowski test suite """
    if solver == "ipopt" and not config.HAS_IPOPT:
        pytest.skip(
            "Test requires optional dependency ipopt, which is not installed.")

    x = sym.MatrixSymbol('x', 2, 1)
    p = sym.Symbol('p')

    prob = OptimizationProblem(mode='min', wrap_using=wrap_using)

    prob.add_parameter(p)
    prob.add_variable(x, lb=[2, 0], ub=[p, p])
    prob.add_constraints_from([x[0] * x[1] >= 25,
                               x[0] ** 2 + x[1] ** 2 >= 25])
    prob.obj = x[0] ** 2 / 100 + x[1] ** 2

    x0 = [2, 2]
    res_50 = prob.solve(x0, 50, solver=solver, tol=tol)
    assert res_50['success']
    assert np.allclose(res_50['x'], np.array([15.8114, 1.58114]))

    res_20 = prob.solve(x0, 20, solver=solver, tol=tol)
    assert res_20['success']
    assert np.allclose(res_20['x'], np.array([15.8114, 1.58114]))


@pytest.mark.parametrize("solver,wrap_using",
                         product(["ipopt", "slsqp"], wrap_using_values))
@needs_ipopt
def test_prob71(solver, wrap_using):
    """ problem 71 from the Hock-Schittkowski test suite """
    if solver == "ipopt" and not config.HAS_IPOPT:
        pytest.skip(
            "Test requires optional dependency ipopt, which is not installed.")

    x = sym.MatrixSymbol('x', 4, 1)
    obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    cons = [x[0] * x[1] * x[2] * x[3] >= 25,
            sym.Eq(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2, 40)]
    lb = np.ones(4)
    ub = 5 * np.ones(4)

    prob_min = OptimizationProblem(mode='min', wrap_using=wrap_using)
    prob_max = OptimizationProblem(mode='max', wrap_using=wrap_using)

    prob_min.add_variable(x, lb=lb, ub=ub)
    prob_min.add_constraints_from(cons)
    prob_min.obj = obj

    prob_max.add_variable(x, lb=lb, ub=ub)
    prob_max.add_constraints_from(cons)
    # test maximization by negating objective
    prob_max.obj = -obj

    x0 = np.array([1, 5, 5, 1])
    res_min = prob_min.solve(x0, solver=solver, tol=tol)
    res_max = prob_max.solve(x0, solver=solver, tol=tol)

    assert res_min['success']
    assert np.allclose(res_min['x'],
                       np.array([1.0, 4.74299964, 3.82114998, 1.37940831]))

    assert res_max['success']
    assert np.allclose(res_max['x'], res_min['x'])
    assert np.allclose(np.abs(res_max['fun']), np.abs(res_min['fun']))


@pytest.mark.parametrize("solver,wrap_using",
                         product(["ipopt", "slsqp"], wrap_using_values))
@needs_ipopt
def test_prob64(solver, wrap_using):
    """ problem 64 from the Hock-Schittkowski test suite """
    x1, x2, x3 = sym.symarray('x', 3)
    p = sym.MatrixSymbol('p', 6, 1)

    prob = OptimizationProblem(wrap_using=wrap_using)
    prob.add_variables_from([x1, x2, x3], lb=1.0e-5)
    prob.add_parameter(p)
    prob.obj = p[0] * x1 + p[1] / x1 + p[2] * x2 + p[3] / x2 + p[4] * x3 +\
        p[5]/x3
    prob.add_constraint(4 / x1 + 32 / x2 + 120 / x3 <= 1)

    x0 = np.ones(3)
    p0 = np.array([5, 50000, 20, 72000, 10, 144000])
    res = prob.solve(x0, p0, solver=solver, tol=tol)

    assert res['success']
    assert np.allclose(res['x'],
                       np.array([108.7347175, 85.12613942, 204.3247078]))


@pytest.mark.parametrize("solver,wrap_using",
                         product(["ipopt", "cobyla", "slsqp"],
                                 wrap_using_values))
@needs_ipopt
def test_prob77(solver, wrap_using):
    """ problem 77 from the Hock-Schittkowski test suite """
    if solver == "ipopt" and not config.HAS_IPOPT:
        pytest.skip(
            "Test requires optional dependency ipopt, which is not installed.")
    x1, x2, x3, x4, x5 = sym.symarray('x', 5)
    prob = OptimizationProblem()
    prob.add_variables_from([x1, x2, x3, x4, x5])

    prob.obj = (x1 - 1) ** 2 + (x1 - x2) ** 2 + (x3 - 1) ** 2 + \
               (x4 - 1) ** 4 + (x5 - 1) ** 6

    # write equality constraints as two inequalities to try cobyla
    cons = [x1 ** 2 * x4 + sym.sin(x4 - x5) - 2 * np.sqrt(2) <= 0,
            x1 ** 2 * x4 + sym.sin(x4 - x5) - 2 * np.sqrt(2) >= 0,
            x2 + x3 ** 4 * x4 ** 2 - 8 - np.sqrt(2) <= 0,
            x2 + x3 ** 4 * x4 ** 2 - 8 - np.sqrt(2) >= 0]
    prob.add_constraints_from(cons)

    x0 = 2 * np.ones(5)
    res = prob.solve(x0, solver=solver, tol=tol)

    assert res['success']
    assert np.allclose(res['x'],
                       np.array([1.166172, 1.182111, 1.380257, 1.506036,
                                 0.6109203]))
