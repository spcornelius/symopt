import numpy as np
import pytest
import sympy as sym

from symopt.problem import OptimizationProblem

tol = 1.0e-8

@pytest.mark.parametrize("method", ["cyipopt", "cobyla", "slsqp"])
def test_prob18(method):
    # problem 18 from Hock-Schittkowski test suite
    x = sym.MatrixSymbol('x', 2, 1)
    p = sym.Symbol('p')

    obj = x[0]**2/100 + x[1]**2
    con = [x[0]*x[1] >= 25,
           x[0]**2 + x[1]**2 >= 25]
    lb = [2, 0]
    ub = [p, p]

    prob = OptimizationProblem(obj, [x], params=[p], constraints=con,
                               lb=lb, ub=ub)
    x0 = [2, 2]
    res_50 = prob.solve(x0, 50, method=method, tol=tol)
    assert res_50['success']
    assert np.allclose(res_50['x'], np.array([15.8114, 1.58114]))

    res_20 = prob.solve(x0, 20, method=method, tol=tol)
    assert res_20['success']
    assert np.allclose(res_20['x'], np.array([15.8114, 1.58114]))


@pytest.mark.parametrize("method", ["cyipopt", "slsqp"])
def test_prob71(method):
    # problem 71 from Hock-Schittkowski test suite
    x = sym.MatrixSymbol('x', 4, 1)
    obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    con = [x[0] * x[1] * x[2] * x[3] >= 25,
           sym.Eq(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2, 40)]

    lb = np.ones(4)
    ub = 5 * np.ones(4)

    prob = OptimizationProblem(obj, [x], constraints=con, lb=lb, ub=ub)
    x0 = np.array([1, 5, 5, 1])
    res = prob.solve(x0, method=method, tol=tol)

    assert res['success']
    assert np.allclose(res['x'],
                       np.array([1.0, 4.74299964, 3.82114998, 1.37940831]))


@pytest.mark.parametrize("method", ["cyipopt", "cobyla", "slsqp"])
def test_prob64(method):
    x1, x2, x3 = sym.symarray('x', 3)
    p = sym.MatrixSymbol('p', 6, 1)
    obj = p[0]*x1 + p[1]/x1 + p[2]*x2 + p[3]/x2 + p[4]*x3 + p[5]/x3
    con = [4/x1 + 32/x2 + 120/x3 <= 1]
    lb = 1.0e-5*np.ones(3)

    prob = OptimizationProblem(obj, [x1, x2, x3], params=[p],
                               constraints=con, lb=lb)
    x0 = np.ones(3)
    p0 = np.array([5, 50000, 20, 72000, 10, 144000])
    res = prob.solve(x0, p0, method=method, tol=tol)

    assert res['success']
    assert np.allclose(res['x'],
                       np.array([108.7347175, 85.12613942, 204.3247078]))
