symopt
======
`symopt` makes it easy to define and solve (non)linear constrained optimization problems in Python. 
It uses the power of [SymPy](https://www.sympy.org/) to automate the error-prone and 
time-consuming task of writing functions to evaluate an optimization problem's objective function 
and nonlinear constraints (to say nothing of their first and second derivatives!). 
`symopt` then provides a standardized interface to solve the problem through nonlinear 
optimization backends including [SciPy](https://www.scipy.org/) and 
[Ipopt](https://projects.coin-or.org/Ipopt).

Documentation
-------------
Auto-generated API documentation for latest stable release can be found at
[ReadTheDocs](https://symopt.readthedocs.io/en/stable/).

Usage
-----
Optimization problems can be defined using the `OptimizationProblem`
class, which has a similar constructor to `scipy.optimize.minimize`.
For example, consider

<p align="center">
    <img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;\textrm{minimize}\;\;&space;&x_1^2/100&space;&plus;&space;x_2^2&space;\\&space;\textrm{subject&space;to}\;\;&space;&&space;x_1&space;x_2&space;\geq&space;25&space;\\&space;&&space;x_1^2&space;&plus;&space;x_2^2&space;\geq&space;25&space;\\&space;&&space;2&space;\leq&space;x_1&space;\leq&space;p_1&space;\\&space;&&space;0&space;\leq&space;x_2&space;\leq&space;p_2&space;\\&space;\end{align*}">
</p>

where the *p*'s  are parameters defining the upper bounds for each variable. 
This can be defined by:
```python
    >>> from symopt import OptimizationProblem
    >>> from sympy import MatrixSymbol, symarray
    
    >>> x1, x2 = symarray('x', 2)
    >>> p = MatrixSymbol('p', 2, 1)
    
    >>> prob = OptimizationProblem(mode='min')
    >>> prob.add_parameter(p)
    >>> prob.add_variable(x1, lb=2, ub=p[0])
    >>> prob.add_variable(x2, lb=0, ub=p[1])
    
    >>> prob.obj = x1**2/100 + x2**2
    >>> prob.add_constraints_from([x1 * x2 >= 25,
                                   x1**2 + x2**2 >= 25])
```
That's it. From here, `symopt` will automatically:

* derive all necessary derivatives (gradients and Hessians for the objective 
function/constraints)
* create functions to numerically evaluate these quantities 
(using SymPy's `lambdify` or `autowrap`)

One can then solve the problem for specified parameters using `solve`:
```python
    >>> x = [2, 2]
    >>> p = [20.0, 50.0]
    >>> res = prob.solve(x, p, method='ipopt')
    >>> print(res['success'])
    True
    >>> print(res['x'])
    array([15.8113883 ,  1.58113883])
    >>> print(res['fun'])
    5.000000000505797
```

Dependencies
------------
* numpy
* scipy
* sympy
* [orderedset](https://pypi.org/project/orderedset/)
* [cyipopt](https://github.com/matthias-k/cyipopt) (Optional, for optimization using the Ipopt backend)
* A fortran compiler (Optional, for code generation using `autowrap`)


License
-------
`symopt` is released under the MIT license. See LICENSE for details.


Author
------
Sean P. Cornelius (gmail address: spcornelius).