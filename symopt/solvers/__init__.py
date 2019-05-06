import symopt.solvers.scipy
import symopt.solvers.ipopt

solve = {'slsqp': scipy.solve_slsqp,
         'cobyla': scipy.solve_cobyla,
         'ipopt': ipopt.solve_ipopt}
