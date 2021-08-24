import math
import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy as np
import scipy

from io import StringIO

from pyomo_opt.common import SOLVER_NAME
from pyomo_opt.common import SOLVER_PATH
from utils.stochastic_search import stochastic_projection


def project(x0, A, b, logger=None, hotstart=None, tol=None):
	n = len(x0)
	m = A.shape[0]
	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	if hotstart is not None:
		if np.all(A @ hotstart <= b + tol):
			opt.options['warm_start_init_point'] = 'yes'
		else:
			hotstart = None
	if hotstart is None and np.all(A @ np.zeros(n) <= b + (tol if tol is not None else 0)):
		hotstart = np.zeros(n)
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
	try:
		result = opt.solve(model, tee=False)
	except:
		print('unable to project...')
		# raise
		# Instead, resort to a simpler, non-pyomo search...
		trial_value = stochastic_projection(x0, A, b, tol)
		if trial_value is None:
			return False, None, None
		else:
			return trial_value.trial is not None, trial_value.trial, trial_value.value

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
