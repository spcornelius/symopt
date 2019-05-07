from itertools import product

import numpy as np
import pytest
import sympy as sym

import symopt.config as config
from symopt.problem import OptimizationProblem

tol = 1.0e-8

wrap_using = ['lambdify', 'autowrap']


@pytest.mark.parametrize("method,wrap_using",
                         product(["ipopt", "slsqp"], wrap_using))
def test_prob18(method, wrap_using):
    """ problem 18 from the Hock-Schittkowski test suite """
    if method == "ipopt" and not config.HAS_IPOPT:
        pytest.skip(
            "Test requires optional dependency ipopt, which is not installed.")

    x = sym.MatrixSymbol('x', 2, 1)
    p = sym.Symbol('p')

    obj = x[0] ** 2 / 100 + x[1] ** 2
    cons = [x[0] * x[1] >= 25,
            x[0] ** 2 + x[1] ** 2 >= 25]
    lb = [2, 0]
    ub = [p, p]

    prob = OptimizationProblem(obj, [x], params=[p], cons=cons,
                               lb=lb, ub=ub, wrap_using=wrap_using)

    x0 = [2, 2]
    res_50 = prob.solve(x0, 50, method=method, tol=tol)
    assert res_50['success']
    assert np.allclose(res_50['x'], np.array([15.8114, 1.58114]))

    res_20 = prob.solve(x0, 20, method=method, tol=tol)
    assert res_20['success']
    assert np.allclose(res_20['x'], np.array([15.8114, 1.58114]))


@pytest.mark.parametrize("method,wrap_using",
                         product(["ipopt", "slsqp"], wrap_using))
def test_prob71(method, wrap_using):
    """ problem 71 from the Hock-Schittkowski test suite """
    if method == "ipopt" and not config.HAS_IPOPT:
        pytest.skip(
            "Test requires optional dependency ipopt, which is not installed.")
    x = sym.MatrixSymbol('x', 4, 1)
    obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    cons = [x[0] * x[1] * x[2] * x[3] >= 25,
            sym.Eq(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2, 40)]

    lb = np.ones(4)
    ub = 5 * np.ones(4)

    prob_min = OptimizationProblem(obj, [x], cons=cons, lb=lb, ub=ub,
                                   wrap_using=wrap_using)
    x0 = np.array([1, 5, 5, 1])
    res_min = prob_min.solve(x0, method=method, tol=tol)

    assert res_min['success']
    assert np.allclose(res_min['x'],
                       np.array([1.0, 4.74299964, 3.82114998, 1.37940831]))

    # test maximization
    prob_max = OptimizationProblem(-obj, [x], cons=cons, lb=lb, ub=ub,
                                   mode="max", wrap_using=wrap_using)
    res_max = prob_max.solve(x0, method=method, tol=tol)
    assert res_max['success']
    assert np.allclose(res_max['x'], res_min['x'])
    assert np.allclose(np.abs(res_max['fun']), np.abs(res_min['fun']))


@pytest.mark.parametrize("method,wrap_using",
                         product(["ipopt", "slsqp"], wrap_using))
def test_prob64(method, wrap_using):
    """ problem 64 from the Hock-Schittkowski test suite """
    x1, x2, x3 = sym.symarray('x', 3)
    p = sym.MatrixSymbol('p', 6, 1)
    obj = p[0] * x1 + p[1] / x1 + p[2] * x2 + p[3] / x2 + p[4] * x3 + p[5] / x3
    cons = [4 / x1 + 32 / x2 + 120 / x3 <= 1]
    lb = 1.0e-5 * np.ones(3)

    prob = OptimizationProblem(obj, [x1, x2, x3], params=[p],
                               cons=cons, lb=lb, wrap_using=wrap_using)
    x0 = np.ones(3)
    p0 = np.array([5, 50000, 20, 72000, 10, 144000])
    res = prob.solve(x0, p0, method=method, tol=tol)

    assert res['success']
    assert np.allclose(res['x'],
                       np.array([108.7347175, 85.12613942, 204.3247078]))


@pytest.mark.parametrize("method,wrap_using",
                         product(["ipopt", "cobyla", "slsqp"], wrap_using))
def test_prob77(method, wrap_using):
    """ problem 77 from the Hock-Schittkowski test suite """
    if method == "ipopt" and not config.HAS_IPOPT:
        pytest.skip(
            "Test requires optional dependency ipopt, which is not installed.")
    x1, x2, x3, x4, x5 = sym.symarray('x', 5)
    obj = (x1 - 1) ** 2 + (x1 - x2) ** 2 + (x3 - 1) ** 2 + (x4 - 1) ** 4 + \
          (x5 - 1) ** 6

    # write equality constraints as two inequalities to try cobyla
    cons = [x1 ** 2 * x4 + sym.sin(x4 - x5) - 2 * np.sqrt(2) <= 0,
            x1 ** 2 * x4 + sym.sin(x4 - x5) - 2 * np.sqrt(2) >= 0,
            x2 + x3 ** 4 * x4 ** 2 - 8 - np.sqrt(2) <= 0,
            x2 + x3 ** 4 * x4 ** 2 - 8 - np.sqrt(2) >= 0]
    prob = OptimizationProblem(obj, [x1, x2, x3, x4, x5], cons=cons,
                               wrap_using=wrap_using)

    x0 = 2 * np.ones(5)
    res = prob.solve(x0, method=method, tol=tol)
    assert res['success']
    assert np.allclose(res['x'],
                       np.array([1.166172, 1.182111, 1.380257, 1.506036,
                                 0.6109203]))
