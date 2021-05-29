import math
import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy as np
import scipy

from io import StringIO

from pyomo_opt.common import SOLVER_NAME
from pyomo_opt.common import SOLVER_PATH


def project(x0, A, b, logger=None, hotstart=None, tol=None):
	n = len(x0)
	m = A.shape[0]
	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	if hotstart is not None:
		opt.options['warm_start_init_point'] = 'yes'
	if tol is not None:
		opt.options['tol'] = tol
	model = ConcreteModel()
	model.x = Var(range(n))

	if hotstart is not None:
		for i in range(n):
			model.x[i].set_value(hotstart[i])

	model.constraints = ConstraintList()
	for i in range(m):
		if np.linalg.norm(A[i]) < 1e-12:
			continue
		model.constraints.add(sum(A[i, j] * model.x[j] for j in range(n)) <= b[i])

	def objective_rule(m):
		return sum((m.x[i] - x0[i]) ** 2 for i in range(n))
	model.objective = Objective(rule=objective_rule, sense=minimize)
	result = opt.solve(model, tee=False)

	if logger is not None:
		string_stream = StringIO()
		model.pprint(ostream=string_stream, verbose=True)
		logger.verbose("Ran projection:\n" + string_stream.getvalue())

	ok = result.solver.status == SolverStatus.ok
	if not ok:
		print("warning solver did not return ok")
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		print("warning solver did not return optimal")

	return ok and optimal, np.array([model.x[i].value for i in range(n)]), np.sqrt(model.objective.expr())
