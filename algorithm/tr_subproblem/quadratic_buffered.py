
import numpy as np

from algorithm.tr_subproblem.tr_solution import TrSolution
from utils.bounds import Bounds
from utils.quadratic import Quadratic
from utils.stochastic_search import StochasticSearchParams, simple_stochastic_search, multi_eval_stochastic_search


class QuadraticBufferedTr:
	def __init__(self):
		self.A = None
		self.b = None
		self.xk = None
		self.r = None
		self.objective = None
		self.buffer_dist = None
		self.buffer_rate = None

		self.plotter = None

		self.z = None
		self.w = None
		self.quad_cons = None
		self.minimizer = None

	def create_constraint(self, ai, wi):
		mat = np.eye(len(ai)) - np.outer(ai, ai)
		sign = np.sign(ai @ (self.xk - wi))
		# func = lambda x: self.buffer_rate * (x - wi) @ mat @ (x - wi) - ai @ (x - wi) * sign

		# self.buffer_rate * (x - wi) @ mat @ (x - wi) - ai @ (x - wi) * sign
		# self.buffer_rate * [x @ mat @ x - 2 * x @ mat @ wi + wi @ mat @ wi] - (ai * sign) @ x + ai @ wi * sign
		quadratic = Quadratic.create(
			self.buffer_rate * wi @ mat @ wi + ai @ wi * sign,
			self.buffer_rate * -2 * mat @ wi - ai * sign,
			self.buffer_rate * mat
		)

		# test_points = np.random.random((100, 2))
		# multi_eval = quadratic.multi_eval(test_points)
		# # print(multi_eval)
		# for v, xtest in zip(multi_eval, test_points):
		# 	f = func(xtest)
		# 	q = quadratic.evaluate(xtest)
		# 	print(f, q, v, abs(f - q) + abs(v - q))

		return quadratic

	def create_constraints(self):
		nrms = np.linalg.norm(self.A, axis=1)
		self.A = self.A / nrms[:, np.newaxis]
		self.b = self.b / nrms

		self.z = self.xk + np.diag(self.b - self.A @ self.xk) @ self.A
		self.w = self.buffer_dist * self.xk + (1 - self.buffer_dist) * self.z
		self.quad_cons = [
			self.create_constraint(ai, wi)
			for ai, wi in zip(self.A, self.w)
		]

	def add_to_plt(self, p):
		p.add_lines(a=self.A, b=self.b, label='Constraint Linearization')
		p.add_linf_tr(center=self.xk, radius=self.r, label='tr', color='k')
		for quad_cons in self.quad_cons:
			p.add_contour(
				quad_cons.evaluate,
				color='m',
				label='quadratic',
				lvls=[-0.1, 0]
			)

		p.add_point(self.xk, label='xk', color='g')
		p.add_points(self.z, label='zeros', color='r')
		p.add_points(self.w, label='vertices', color='g')
		if self.minimizer is not None:
			p.add_point(self.minimizer, label='minimizer', color='b', marker='o')

	def create_plot(self):
		bounds = Bounds() \
			.extend_tr(self.xk, self.r) \
			.expand(1.2)
		p = self.plotter.create_plot(
			'quadratic_tr',
			bounds=bounds,
			title='quadratic buffered tr',
			subfolder='quad_tr_subproblems')
		self.add_to_plt(p)
		p.save()

	def stochastic_obj(self, x):
		return (
			np.linalg.norm(x - self.xk, ord=np.inf) <= self.r and
			np.all([q.evaluate(x) <= 0 for q in self.quad_cons]),
			self.objective.evaluate(x),
		)

	def stochastic_multi_eval(self, x):
		obj = self.objective.multi_eval(x)
		feas = np.linalg.norm(x - self.xk, axis=1, ord=np.inf) <= self.r
		for q in self.quad_cons:
			feas = np.logical_and(feas, q.multi_eval(x) <= 0)
		obj[np.logical_not(feas)] = np.nan
		return obj


def search_quadratic_buffered(state, model, br, solutions):
	if np.sum(br.active_indices) == 0:
		return

	qbtr = QuadraticBufferedTr()
	qbtr.objective = model.shifted_objective
	qbtr.A = model.shifted_A[br.active_indices]
	qbtr.b = model.shifted_b[br.active_indices]
	qbtr.xk = model.shifted_x
	qbtr.r = model.shifted_r
	qbtr.plotter = state.plotter
	qbtr.buffer_dist = 1 - br.adpa
	qbtr.buffer_rate = br.bdpb

	qbtr.create_constraints()

	ssp = StochasticSearchParams()
	ssp.x0 = qbtr.xk
	ssp.objective = qbtr.stochastic_obj
	ssp.multi_eval = qbtr.stochastic_multi_eval
	ssp.initial_radius = qbtr.r / 2.0

	trial_value = multi_eval_stochastic_search(ssp)

	solutions.append(TrSolution.create(
		TrSolution.Types.HEURISTIC,
		'quadratic-buffered',
		trial_value
	))

	qbtr.minimizer = trial_value.trial
	qbtr.create_plot()




