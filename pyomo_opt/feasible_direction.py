
import math
import pyomo.environ

from io import StringIO

from pyomo.core import *
from pyomo.opt import *
import numpy as np

from pyomo_opt.common import SOLVER_NAME
from pyomo_opt.common import SOLVER_PATH


def find_feasible_direction(gradients, logger, tol=None):
	n = gradients.shape[1]

	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	model = ConcreteModel()
	model.u = Var(range(n))
	model.t = Var()

	# if hotstart is not None:
	# 	opt.options['warm_start_init_point'] = 'yes'
	# 	hotstart.apply(model)
	if tol is not None:
		opt.options['tol'] = tol

	for i in range(n):
		model.u[i].setlb(-1)
		model.u[i].setub(1)

	norms = np.linalg.norm(gradients, axis=1)
	grads = -gradients / norms[:, np.newaxis]

	model.constraints = ConstraintList()
	for grad in grads:
		model.constraints.add(
			model.t <= sum(grad[i] * model.u[i] for i in range(n))
		)
	model.constraints.add(sum(model.u[i] ** 2 for i in range(n)) <= 1)

	def objective_rule(m):
		return m.t

	model.objective = Objective(rule=objective_rule, sense=maximize)

	try:
		result = opt.solve(model, tee=False)
	except:
		print('here')
		return False, None, None

	string_stream = StringIO()
	model.pprint(ostream=string_stream, verbose=True)
	if logger is not None:
		logger.verbose("Computed feasible direction:\n" + string_stream.getvalue())

	ok = result.solver.status == SolverStatus.ok
	if not ok:
		print("warning solver did not return ok")
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		print("warning solver did not return optimal")

	u = np.array([model.u[i].value for i in range(n)])
	t = model.objective.expr()
	return t > 0, u, t
