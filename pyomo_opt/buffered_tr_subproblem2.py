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


class TrustRegionSubproblem:
	def __init__(self):
		self.m = None
		self.n = None
		self.x = None
		self.radius = None
		self.ws = None
		self.beta = None
		self.objective = None
		self.unit_gradients = None
		self.hotstart = None


def solve_buffered_tr_subproblem(br_model, logger):
	n = br_model.n
	m = br_model.m

	opt = SolverFactory(SOLVER_NAME, executable=SOLVER_PATH)
	model = ConcreteModel()
	model.x = Var(range(n))

	for i in range(n):
		model.x[i].setlb(br_model.x[i] - br_model.radius)
		model.x[i].setub(br_model.x[i] + br_model.radius)

	if br_model.hotstart is not None:
		opt.options['warm_start_init_point'] = 'yes'
		br_model.hotstart.apply(model, use_st=False)
	if br_model.tol is not None:
		opt.options['tol'] = br_model.tol

	model.constraints = ConstraintList()

	for i in range(m):
		lhs = sum(
			(model.x[j] - br_model.ws[i, j]) * (-br_model.unit_gradients[i, j])
			for j in range(n)
		)
		rhs = br_model.beta ** 2 * sum(
			(model.x[j] - br_model.ws[i, j]) ** 2
			for j in range(n)
		)
		model.constraints.add(lhs >= 0)
		model.constraints.add(lhs ** 2 >= rhs)

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
