======
symopt
======

Easy (non)linear optimization with symbolically-defined
objectives/constraints, with automatic calculation of derivatives.
Uses SciPy_ and Ipopt_ (through cyipopt_) as optimization backends.

.. _Ipopt: https://projects.coin-or.org/Ipopt
.. _SciPy: https://www.scipy.org/
.. _cyipopt: https://github.com/matthias-k/cyipopt

Usage
-----

Optimization problems can be defined using the `OptimizationProblem`
class, which has a similar constructor to `scipy.optimize.minimize`.
For example, consider

.. math::
    \text{minimize}\; &x_1^2/100 + x_2^2 \\
    \text{s.t.}\; &x_1 x_2 \geq 25 \\
                  &x_1^2 + x_2^2 \geq 25 \\
                  &2 \leq x_1 \leq p_1 \\
                  &0 \leq x_2 \leq p_2 \\

where :math:`p_1` and :math:`p_2` are parameters defining
the upper bounds for each variable. This can be defined
by:

.. code:: python

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

That's it. From here, `symopt` will automatically create the corresponding functions to
numerically evaluate the objective, constraints, and upper/lower bounds, as well
as those of the relevant derivatives (e.g. objective and constraint gradients). One can then solve the problem for specified parameters using `solve`:

.. code:: python

    x0 = [2, 2]
    p = [20.0, 50.0]
    res = prob.solve(x0, p, method='cyipopt')

Note that the objective function, constraints, and upper/lower
bounds can depend on the declared parameters. Variables and parameters
may be a mixture of `Symbol` and `MatrixSymbol` objects.

Dependencies
------------
* numpy
* scipy
* sympy
* orderedset_

.. _orderedset: https://pypi.org/project/orderedset/

Optional dependencies
---------------------
* cyipopt_ (for optimizatoin using the IPOPT backend)


.. _cyipopt: https://github.com/matthias-k/cyipopt

License
-------
``symopt`` is released under the MIT license. See LICENSE for details.