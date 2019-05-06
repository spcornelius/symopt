import sympy as sym
from sympy.utilities.autowrap import autowrap

from symopt.util import chain_scalars, squeezed, is_linear, is_quadratic, \
    depends_only_on


class SymOptBase(object):
    """ Base class for symbolic expression with automatic derivatives
        and function wrapping.

        Attributes
        ----------
        expr : `~sympy.core.expr.Expr`
            Relevant symbolic expression in a child class, i.e. the thing
            that should have derivatives/function wrapping.
        vars : `~collections.abc.Sequence` of `~typing.Union` \
                [`~sympy.core.symbol.Symbol`,\
                 `~sympy.matrices.expressions.MatrixSymbol` ]
            Symbolic variables.
        params : `~collections.abc.Sequence` of `~typing.Union` \
                [`~sympy.core.symbol.Symbol`,\
                 `~sympy.matrices.expressions.MatrixSymbol` ]
            Symbolic parameters.
        grad : `~sympy.matrices.immutable.ImmutableDenseMatrix`
            Derivatives of :py:attr:`expr` with respect to \
            :py:attr:`vars`.
        hess : `~sympy.matrices.immutable.ImmutableDenseMatrix`
            Second derivatives of py:attr:`expr` with respect to \
            :py:attr:`vars`.
        cb : `~collections.abc.Callable`
            Function to numerically evaluate :py:attr:`expr`. Has signature
            ``cb(*vars, *params) -->`` `float`.
        grad_cb : `~collections.abc.Callable`
            Function to numerically evaluate :py:attr:`grad`. Has signature
            ``grad_cb(*vars, *params) -->`` 1D `numpy.ndarray`.
        hess_cb : `~collections.abc.Callable`
            Function to numerically evaluate :py:attr:`hess`. Has signature
            ``hess_cb(*vars, *params) -->`` 2D `numpy.ndarray`.
        """

    def __init__(self, expr, vars, params):
        """ Base class for symbolic expression with automatic derivatives
        and function wrapping.

        Parameters
        ----------
        expr : `~sympy.core.expr.Expr`
            Relevant symbolic expression, as specified by a child class.
            I.e. the thing that should have derivatives/function wrapping.
        vars : `~collections.abc.Sequence` of `~typing.Union` \
                [`~sympy.core.symbol.Symbol`,\
                 `~sympy.matrices.expressions.MatrixSymbol` ]
            Symbolic variables. Note: not all :py:attr:`vars` need appear in
            :py:attr:`expr`; they merely define over what to take derivatives
            of :py:attr:`expr`, and the signature of the functions to
            numerically evaluate it and its derivatives.
        params : `~collections.abc.Sequence` of `~typing.Union` \
                [`~sympy.core.symbol.Symbol`,\
                 `~sympy.matrices.expressions.MatrixSymbol` ]
            Parameters appearing in :py:attr:`expr`. Note: not all \
            :py:attr:`params` need appear in :py:attr:`expr`.
        """

        try:
            expr = sym.sympify(expr)
        except sym.SympifyError:
            raise TypeError(f"Couldn't sympify expression '{expr}.")
        params_and_vars = set(params) | set(vars)
        if not depends_only_on(expr, params_and_vars):
            raise ValueError(
                f"Expression '{expr}' can depend only on "
                f"declared vars and params.")
        self.expr = expr
        self.vars = vars
        self.params = params

        scalar_vars = list(chain_scalars(vars))
        self.grad = sym.ImmutableMatrix([expr.diff(s) for s in scalar_vars])
        self.hess = sym.ImmutableMatrix(self.grad.jacobian(scalar_vars))

        x = sym.MatrixSymbol('x', len(scalar_vars), 1)
        subs = dict(zip(scalar_vars, sym.flatten(x)))
        args = (x,) + tuple(self.params)

        def _autowrap(_expr):
            return autowrap(_expr.subs(subs), args=args)

        self.cb = _autowrap(self.expr)
        self.grad_cb = squeezed(_autowrap(self.grad))
        self.hess_cb = squeezed(_autowrap(self.hess))

    def is_linear(self):
        """ Return `True` if :py:attr:`self.expr` is linear in
            :py:attr:`self.vars`, `False` otherwise."""
        return is_linear(self.expr, self.vars)

    def is_quadratic(self):
        """ Return `True` if :py:attr:`self.expr` is at most quadratic
            in :py:attr:`self.vars`, `False` otherwise."""
        return is_quadratic(self.expr, self.vars)