import sympy as sym
import abc
from sympy import lambdify, sympify, Symbol, MatrixSymbol
from sympy.utilities.autowrap import autowrap

from symopt.util import is_linear, is_quadratic, reshape_args, squeezed, \
    is_scalar, as_scalars

__all__ = []
__all__.extend([
    'SymOptSymbol',
    'SymOptExpr'
])


class SymOptExpr(object, metaclass=abc.ABCMeta):
    """ Base class for symbolic expression with automatic derivatives
        and function wrapping.

        Attributes
        ----------
        prob: `.OptimizationProblem`
            The containing optimization problem.
        """

    def __init__(self, prob, wrap_using='autowrap', simplify=True):
        """ Base class for symbolic expression with automatic derivatives
        and function wrapping.

        Parameters
        ----------
        prob: `.OptimizationProblem`
            The containing optimization problem.
        wrap_using : `str`, either 'lambdify' or 'autowrap'
            Which backend to use for wrapping the
            constraint and its derivatives for numerical
            evaluation. See :func:`~sympy.utilities.lambdify.lambdify` and
            :func:`~sympy.utilities.autowrap.autowrap` for more details.
            Defaults to 'lambdify'.
        simplify : `bool`
            If `True`, simplify constraint and its derivatives
            before wrapping as functions. Defaults to `True`.

        """
        self.prob = prob
        if not prob.depends_only_on_params_or_vars(self.expr):
            raise ValueError(
                f"Expression '{self.expr}' can depend only on "
                f"problem variables and parameters.")
        self._grad = None
        self._hess = None
        self._cb = None
        self._grad_cb = None
        self._hess_cb = None
        self.simplify = simplify
        self.wrap_using = str(wrap_using).lower()

    def _wrap(self, expr):
        vars = self.prob.var_scalars
        x = sym.MatrixSymbol('x', len(vars), 1)
        subs = dict(zip(vars, sym.flatten(x)))
        expr = expr.subs(subs)
        args = (x,) + tuple(p.symbol for p in self.prob.params)
        if self.simplify:
            expr = expr.simplify()
        if self.wrap_using == 'autowrap':
            return autowrap(expr, args=args)
        else:
            return reshape_args(lambdify(args, expr),
                                args)

    @property
    @abc.abstractmethod
    def expr(self):
        """ The symbolic expression (and its derivatives) that should be \
            wrapped in functions. """

    @property
    @abc.abstractmethod
    def sympified(self):
        """ Attribute to return upon calls to \
            :func:`~sympy.core.sympify.sympify` applied to this object."""

    def _sympy_(self):
        return self.sympified

    def __eq__(self, other):
        return sympify(self).equals(other)

    def __ne__(self, other):
        return not sympify(self).equals(other)

    def __str__(self):
        return str(sympify(self))

    def __hash__(self):
        # Can't do hash(sympify(self)) due to infinite recursion
        return hash(self.sympified)

    @property
    def grad(self):
        """ `~sympy.matrices.immutable.ImmutableDenseMatrix` : \
             First derivatives of :py:attr:`expr` with respect to \
             :py:attr:`prob.vars`. """
        if self._grad is None:
            self._grad = sym.ImmutableMatrix(
                [self.expr.diff(s) for s in self.prob.var_scalars])
        return self._grad

    @property
    def hess(self):
        """ `~sympy.matrices.immutable.ImmutableDenseMatrix` : \
             Second derivatives of :py:attr:`expr` with respect to \
             :py:attr:`prob.vars`. """
        if self._hess is None:
            self._hess = sym.ImmutableMatrix(
                self.grad.jacobian(self.prob.var_scalars))
        return self._hess

    def cb(self, x, *params):
        """ Callback for numerically evaluating :py:attr:`self.expr`. """
        if self._cb is None:
            self._cb = self._wrap(self.expr)

        return self._cb(x, *params)

    def grad_cb(self, x, *params):
        """ Callback for numerically evaluating :py:attr:`self.grad`. """
        if self._grad_cb is None:
            self._grad_cb = squeezed(self._wrap(self.grad))

        return self._grad_cb(x, *params)

    def hess_cb(self, x, *params):
        """ Callback for numerically evaluating :py:attr:`self.hess`. """
        if self._hess_cb is None:
            self._hess_cb = squeezed(self._wrap(self.hess))

        return self._hess_cb(x, *params)

    def is_linear(self):
        """ Return `True` if :py:attr:`self.expr` is linear in
            :py:attr:`self.vars`, `False` otherwise."""
        return is_linear(self.expr, self.prob.vars)

    def is_quadratic(self):
        """ Return `True` if :py:attr:`self.expr` is at most quadratic
            in :py:attr:`self.vars`, `False` otherwise."""
        return is_quadratic(self.expr, self.prob.vars)


class SymOptSymbol(object):
    """ Base class for variables and parameters.

        Wraps a `~sympy.core.symbols.Symbol` or
        `~sympy.matrices.expressions.MatrixSymbol`, providing additional
        functionality.
    """

    def __init__(self, symbol, prob):
        """ Base class for variables and parameters.

        Parameters
        ----------
        symbol: `~sympy.core.symbols.Symbol` or \
                    `~sympy.matrices.expressions.MatrixSymbol`
            The symbol representing the variable or parameter.
        prob: `.OptimizationProblem`
            The containing optimization problem.
        """
        self.symbol = sympify(symbol)
        self.prob = prob
        if not isinstance(self.symbol, (Symbol, MatrixSymbol)):
            raise TypeError(
                f"Could not convert {symbol} to a Symbol or MatrixSymbol.")

    @property
    def is_scalar(self):
        """`bool`: `True` if :py:attr:`self` represents a scalar symbol,
                   `False` otherwise. """
        return is_scalar(self.symbol)

    def as_scalars(self):
        return as_scalars(self.symbol)

    def _sympy_(self):
        return self.symbol

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other):
        return self.symbol.equals(other)

    def __ne__(self, other):
        return not self.symbol.equals(other)
