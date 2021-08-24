
import numpy as np

from algorithm.tr_subproblem.tr_solution import TrSolution
from utils.bounds import Bounds
from utils.quadratic import Quadratic
from utils.stochastic_search import StochasticSearchParams, simple_stochastic_search, multi_eval_stochastic_search


def add_to_plt(p, model, penalty_buffered_constraints, minimizer=None):
	p.add_lines(a=model.shifted_A, b=model.shifted_b, label='Constraint Linearization')
	p.add_linf_tr(center=model.shifted_x, radius=model.shifted_r, label='tr', color='k')
	for idx, quad_cons in enumerate(model.shifted_constraints):
		p.add_contour(
			quad_cons.evaluate,
			color='m',
			label='constraint ' + str(idx),
			lvls=[-0.1, 0]
		)
	for idx, quad_cons in enumerate(penalty_buffered_constraints):
		p.add_contour(
			quad_cons.evaluate,
			color='r',
			label='penalty buffered constraint ' + str(idx),
			lvls=[-0.1, 0]
		)

	p.add_point(model.shifted_x, label='xk', color='g')
	if minimizer is not None:
		p.add_point(minimizer, label='minimizer', color='b', marker='o')


def create_plot(state, model, penalty_buffered_constraints, minimizer, name):
	bounds = Bounds() \
		.extend_tr(model.shifted_x, model.shifted_r) \
		.expand(1.2)
	p = state.plotter.create_plot(
		name.replace('-', '_'),
		bounds=bounds,
		title='penalty buffered tr',
		subfolder='penalty_tr_subproblems')
	add_to_plt(p, model, penalty_buffered_constraints, minimizer)
	p.save()


def get_stochastic_objective(model, penalty_buffered_constraints):
	def stochastic_obj(x):
		return (
			np.linalg.norm(x - model.shifted_x, ord=np.inf) <= model.shifted_r and
			np.all([q.evaluate(x) <= 1e-8 for q in penalty_buffered_constraints]),
			model.shifted_objective.evaluate(x),
		)
	return stochastic_obj


def get_stochastic_multi_eval(model, penalty_buffered_constraints):
	def stochastic_multi_eval(x):
		obj = model.shifted_objective.multi_eval(x)
		feas = np.linalg.norm(x - model.shifted_x, axis=1, ord=np.inf) <= model.shifted_r
		for q in penalty_buffered_constraints:
			feas = np.logical_and(feas, q.multi_eval(x) <= 0)
		obj[np.logical_not(feas)] = np.nan
		return obj
	return stochastic_multi_eval


def convexify(constraint):
	l, v = np.linalg.eig(constraint.Q)
	return Quadratic.create(
		c=constraint.c,
		g=constraint.g.copy(),
		Q=v @ np.diag(np.maximum(0, l)) @ v.T
	)


def search_penalty_buffered(state, model, br, solutions):
	if np.sum(br.active_indices) == 0:
		return

	penalty = Quadratic.create(
		c=0,
		g=np.zeros(model.n),
		Q=br.bdpb * model.r ** 2 * np.eye(model.n))

	for name, buffered_constraints in [
		('penalty-buffered',
			[constraint + penalty
				for constraint in model.shifted_constraints]),
		('convex-penalty-buffered',
			[convexify(constraint) + penalty
				for constraint in model.shifted_constraints])]:
		ssp = StochasticSearchParams()
		ssp.x0 = model.shifted_x
		ssp.objective = get_stochastic_objective(model, buffered_constraints)
		ssp.multi_eval = get_stochastic_multi_eval(model, buffered_constraints)
		ssp.initial_radius = model.shifted_r / 2.0

		trial_value = multi_eval_stochastic_search(ssp)

		solutions.append(TrSolution.create(
			TrSolution.Types.HEURISTIC,
			name,
			trial_value
		))

		create_plot(state, model, buffered_constraints, trial_value.trial, name)




