Overview
========

Easy (non)linear optimization with symbolically-defined
objectives/constraints, with automatic calculation of derivatives.
Uses [SciPy](https://www.scipy.org/) and [Ipopt](https://projects.coin-or.org/Ipopt) 
(through [cyipopt](https://github.com/matthias-k/cyipopt)) as optimization backends.

Usage
-----

Optimization problems can be defined using the `OptimizationProblem`
class, which has a similar constructor to `scipy.optimize.minimize`.
For example, consider

<p align="center">
    <img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\textrm{minimize}\;\;&space;&x_1^2/100&space;&plus;&space;x_2^2&space;\\&space;\textrm{subject&space;to}\;\;&space;&&space;x_1&space;x_2&space;\geq&space;25&space;\\&space;&&space;x_1^2&space;&plus;&space;x_2^2&space;\geq&space;25&space;\\&space;&&space;2&space;\leq&space;x_1&space;\leq&space;p_1&space;\\&space;&&space;0&space;\leq&space;x_2&space;\leq&space;p_2&space;\\&space;\end{align*}">
</p>

where the *p*'s  are parameters defining the upper bounds for each variable. This can be defined
by:
```python
    from symopt import OptimizationProblem
    from sympy import MatrixSymbol, symbols

    x1, x2 = symbols('x_1 x_2')
    p = MatrixSymbol('p')

    obj = x1**2/100 + x2**2
    con = [x1 * x2 >= 25,
           x1**2 + x2**2 >= 25]
    lb = [2, 0]
    ub = [p[0], p[1]]

    prob = OptimizationProblem(obj, [x], params=[p], constraints=con,
                               lb=lb, ub=ub, mode='min')
```
That's it. From here, `symopt` will automatically create the corresponding functions to
numerically evaluate the objective, constraints, and upper/lower bounds, as well
as those of the relevant derivatives (e.g. objective and constraint gradients). One can then solve the problem for specified parameters using `solve`:
```python
    x0 = [2, 2]
    p = [20.0, 50.0]
    res = prob.solve(x0, p, method='cyipopt')
```
Note that the objective function, constraints, and upper/lower
bounds can depend on the declared parameters. Variables and parameters
may be a mixture of `Symbol` and `MatrixSymbol` objects.

Dependencies
------------
* numpy
* scipy
* sympy
* [orderedset](https://pypi.org/project/orderedset/)

Optional dependencies
---------------------
* [cyipopt](https://github.com/matthias-k/cyipopt) (for optimizationn using the IPOPT backend)


License
-------
`symopt` is released under the MIT license. See LICENSE for details.


Author
------
Sean P. Cornelius (gmail address: spcornelius).
