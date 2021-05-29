import math
import pyomo.environ
from pyomo.core import *
from pyomo.opt import *
import numpy as np
import scipy

from io import StringIO

from pyomo_opt.common import SOLVER_NAME
from pyomo_opt.common import SOLVER_PATH
from utils.polyhedron import Polyhedron
from pyomo_opt.project import project


def solve_buffered_tr_subproblem(br_model, logger):
	n = br_model.n
	m = br_model.m

	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	model = ConcreteModel()
	model.x = Var(range(n))
	model.s = Var(range(m), range(n))
	model.t = Var(range(m))

	for i in range(m):
		model.t[i].setlb(0)
	for i in range(m):
		for j in range(n):
			model.s[i, j].setlb(-1.0)
			model.s[i, j].setub(+1.0)

	if br_model.hotstart is not None:
		opt.options['warm_start_init_point'] = 'yes'
		br_model.hotstart.apply(model)
	if br_model.tol is not None:
		opt.options['tol'] = br_model.tol

	model.constraints = ConstraintList()
	for i in range(m):
		for j in range(n):
			model.constraints.add(model.x[j] == br_model.ws[i, j] + model.t[i] * model.s[i, j])

	for i in range(m):
		model.constraints.add(sum(model.s[i, j] ** 2 for j in range(n)) == 1)

	for i in range(n):
		model.constraints.add(inequality(br_model.x[i] - br_model.radius, model.x[i], br_model.x[i] + br_model.radius))

	for i in range(m):
		model.constraints.add(-sum(model.s[i, j] * br_model.unit_gradients[i, j] for j in range(n)) >= br_model.beta)

	def objective_rule(m):
		return br_model.objective.to_pyomo(m.x)

	model.objective = Objective(rule=objective_rule, sense=minimize)
	result = opt.solve(model, tee=False)

	string_stream = StringIO()
	model.pprint(ostream=string_stream, verbose=True)
	logger.verbose("Ran trust region subproblem:\n" + string_stream.getvalue())

	ok = result.solver.status == SolverStatus.ok
	if not ok:
		print("warning solver did not return ok")
	optimal = result.solver.termination_condition == TerminationCondition.optimal
	if not optimal:
		print("warning solver did not return optimal")

	return ok and optimal, np.array([model.x[i].value for i in range(n)]), model.objective.expr()
