
from qpsolvers import solve_qp, Problem, solve_problem

qp_problem = Problem.load('not_optimal.npz')
qp_problem.check_constraints()
# qp_problem.cond()
# solution = solve_problem(qp_problem, solver="daqp", verbose=True)
solution = solve_problem(qp_problem, solver="daqp", dual_tol=1e-8, primal_tol=1e-8)
# solution = solve_problem(qp_problem, solver="clarabel")
# solution = solve_problem(qp_problem, solver="clarabel", 
#                          tol_feas=1e-9, tol_gap_abs=1e-9, tol_gap_rel=0)
# solution = solve_problem(qp_problem, solver="scs")
sol = solution.x

print(solution)
print(solution.is_optimal(1e-8))
# print(solution.is_optimal(1e-0))
