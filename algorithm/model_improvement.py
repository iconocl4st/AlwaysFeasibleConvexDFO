import numpy as np

from utils.assertions import make_assertion
from utils.default_stringable import DefaultStringable
from utils.ellipsoid import Ellipsoid
from utils.lagrange import LagrangeParams, perform_lu_factorization
from utils.polyhedron import Polyhedron
from utils.quadratic import Quadratic


class Model(DefaultStringable):
	def __init__(self):
		self.sample = None

		self.unshifted_objective = None
		self.shifted_objective = None
		self.unshifted_constraints = None
		self.shifted_constraints = None

		self.x = None
		self.shifted_x = None

		self.r = None
		self.shifted_r = None

		self.unshifted_gradient = None
		self.gradient_of_shifted = None
		self.shifted_negative_grad = None

		self.unshifted_A = None
		self.unshifted_b = None
		self.shifted_A = None
		self.shifted_b = None

	@property
	def n(self):
		return self.sample.shape[1]

	def shift(self, point):
		return (point - self.x) * min(1e5, 1.0 / self.r)

	def get_shifted_tr_constraints(self):
		return Polyhedron.create_from_tr(np.zeros(self.n), 1.0)

	def get_unshifted_tr_constraints(self):
		return Polyhedron.create_from_tr(self.x, self.r)

	def get_shifted_model_constraints(self):
		return Polyhedron.create(self.shifted_A, self.shifted_b)

	def get_unshifted_model_constraints(self):
		return Polyhedron.create(self.unshifted_A, self.unshifted_b)

	def get_all_unshifted_constraints(self):
		return self.get_unshifted_model_constraints().intersect(self.get_unshifted_tr_constraints())

	def get_all_shifted_constraints(self):
		return self.get_shifted_model_constraints().intersect(self.get_shifted_tr_constraints())

	def to_json(self):
		return {
			'sample-points': self.sample,
			'shifted-objective': self.shifted_objective,
			'unshifted-objective': self.unshifted_objective,
			'shifted-constraints': self.shifted_constraints,
			'unshifted-constraints': self.unshifted_constraints,
			'unshifted-center': self.x,
			'shifted-center': self.shifted_x,
			'unshifted-radius': self.r,
			'shifted-radius': self.shifted_r,
			'unshifted-gradient': self.unshifted_gradient,
			'gradient-of-shifted-objective': self.gradient_of_shifted,
			'shifted-negative-gradient': self.shifted_negative_grad,
			'unshifted-A': self.unshifted_A,
			'shifted-A': self.shifted_A,
			'unshifted-b': self.unshifted_b,
			'shifted-b': self.shifted_b,
		}


def improve_geometry(state):
	# state.sample_region = Ellipsoid.create(
	# 	np.array([[3, 1], [0, 1]]),
	# 	state.current_iterate + 1e-4 * np.array([2, 1]) / 2,
	# 	1e-4
	# )

	state.logger.start_step('improving geometry')
	filter_method = {
		'trust-region-radius': lambda x: (
			np.linalg.norm(x - state.current_iterate) < 2.5 * state.outer_tr_radius or
			state.sample_region.contains(x)
		)
	}[state.params.sample_point_filter]

	lparams = LagrangeParams()
	lparams.plotter = state.plotter
	lparams.basis = state.basis
	lparams.xsi_replace = state.params.xsi_replace
	lparams.sample_region = state.sample_region
	lparams.evaluations = state.history.get_evaluations(filter_method)
	# This ensures the current iterate is first, so that it will be present in the model
	lparams.evaluations = sorted(lparams.evaluations, key=lambda e: np.linalg.norm(e[1].x - state.current_iterate))
	lparams.plot_maximizations = state.params.plot_lagrange_maximizations
	lparams.logger = state.logger

	state.logger.verbose(
		'Performing LU factorization on ' + str(len(lparams.evaluations)) + ' points.')
	certification = perform_lu_factorization(lparams)
	make_assertion(certification.success, 'Unable to improve geometry')

	state.logger.stop_step()
	return certification


def create_sr_shifted_models(state, cert, sr):
	n = state.dim
	f_quad = Quadratic.zeros(n)
	c_quads = [Quadratic.zeros(n) for _ in range(state.num_constraints)]
	sample = []
	for idx, unshifted, quad in cert.shifted_info():
		point = sr.unshift_point(unshifted)
		if idx < 0:
			evaluation = state.evaluate(point)
			if not evaluation.has_information():
				state.logger.info('attempted to evaluate infeasible sample point: ' + str(evaluation.x))
				state.logger.stop_step()
				return False, None, None, None
		else:
			evaluation = state.history.get_evaluation(idx)
			make_assertion(
				np.linalg.norm(point - evaluation.x) / max(1.0, np.linalg.norm(point)) < 1e-12,
				'incorrect index of evaluation')
			make_assertion(evaluation.has_information(), 'attempting to use evaluation with no information')
		sample.append(evaluation)

		f_quad += quad * evaluation.objective
		for i in range(state.num_constraints):
			c_quads[i] += quad * evaluation.constraints[i]
	return True, sample, f_quad, c_quads


def update_model(state, cert):
	state.logger.start_step('Updating model')
	sr = state.sample_region
	n = state.dim
	zeros = np.zeros(n)

	success, sample, f_quad, c_quads = create_sr_shifted_models(state, cert, sr)
	if not success:
		return None

	model = Model()
	model.r = state.outer_tr_radius
	model.shifted_r = 1.0
	model.x = state.current_iterate
	model.shifted_x = zeros

	model.sample = np.array([e.x for e in sample])
	model.unshifted_objective = f_quad.compose(sr.l, sr.center, sr.r)
	model.unshifted_constraints = [
		c_quad.compose(sr.l, sr.center, sr.r) for c_quad in c_quads]

	model.shifted_objective = f_quad.compose(
		sr.l, (sr.center - model.x) / model.r, sr.r / model.r)
	model.shifted_constraints = [
		c_quad.compose(sr.l, (sr.center - model.x) / model.r, sr.r / model.r)
		for c_quad in c_quads]

	# tr = model

	# s = np.random.random(2)
	# x = model.x + model.r * s
	# print('s=', s)
	# print('x=', x)
	# print('u=', sr.l @ (x - sr.center) / sr.r)
	# print('u=', sr.l @ (tr.x + tr.r * s - sr.center) / sr.r)
	# print('u=', sr.l @ (tr.r * s - (sr.center - tr.x)) / sr.r)
	# print('u=', sr.l @ (s - (sr.center - tr.x) / tr.r) / (sr.r / tr.r))
	# u = sr.l @ (s - (sr.center - tr.x) / tr.r) / (sr.r / tr.r)
	# print('x=', sr.center + sr.r * sr.linv @ u)
	#
	# print(f_quad.evaluate(u))
	# print(model.unshifted_objective.evaluate(x))
	# print(model.shifted_objective.evaluate(s))

	# u = sr.l @ (x - sr.c) / sr.r
	# s = (x - tr.c) / tr.r
	# 	x = tr.c + tr.r * s
	# u = sr.l @ (tr.c + tr.r * s - sr.c) / sr.r
	# u = sr.l @ (tr.r * s - (sr.c - tr.c)) / sr.r
	# u = sr.l @ (s - (sr.c - tr.c) / tr.r) / (sr.r / tr.r)

	# u = sr.l @ (x - sr.c) / sr.r
	# ==> x = sr.c + sr.r * sr.linv @ u
	# s = (x - tr.c) / tr.r
	# ==> s = (sr.c + sr.r * sr.linv @ u - tr.c) / tr.r
	# ==> s = sr.r * (sr.linv @ u - (tr.c - sr.c) / sr.r) / tr.r
	# ==> s = (sr.linv @ u - (tr.c - sr.c) / sr.r) / (tr.r / sr.r)
	# ==> s = (sr.r / tr.r) * sr.linv @ u - (tr.c - sr.c) / tr.r

	# f1 = c + b @ s + s @ q @ s
	# \nabla f1 = b + 2 * q @ s
	# f2 = c + b @ (l @ (x - c)) / r + (l @ (x - c)) @ q @ (l @ (x - c)) / r ** 2
	# \nabla f2 = b @ l / r + 2 l.T @ q @ l @ (x - c) / r ** 2


	# sr_gradient = f_quad.evaluate_gradient(np.zeros(n))
	# unshifted_gradient1 = model.unshifted_objective.evaluate_gradient(state.current_iterate)
	# quad = f_quad
	# unshifted_gradient2 = quad.g @ sr.l / sr.r + 2 * sr.l.T @ quad.Q @ sr.l @ (state.current_iterate - sr.center) / sr.r ** 2
	# f3 = c + b @ (x - c) / r + (x - c) @ q @ (x - c) / r ** 2

	model.unshifted_gradient = model.unshifted_objective.evaluate_gradient(state.current_iterate)
	model.gradient_of_shifted = model.shifted_objective.evaluate_gradient(zeros)

	model.shifted_negative_grad = model.shift(model.x - model.unshifted_gradient)

	model.shifted_A = np.array([ci.evaluate_gradient(zeros) for ci in model.shifted_constraints])
	model.grad_norms = np.linalg.norm(model.shifted_A, axis=1)
	model.shifted_A /= model.grad_norms[:, np.newaxis]
	model.shifted_b = np.array([
		-ci.evaluate(zeros) / model.grad_norms[i]
		for i, ci in enumerate(model.shifted_constraints)])

	# 		A @ (x - center) / radius <= b
	# ==>	A @ x <= radius * b + A @ center
	model.unshifted_b = model.r * model.shifted_b + model.shifted_A @ model.x
	model.unshifted_A = model.shifted_A

	ensure_model_accuracy(state, model, sample, f_quad, c_quads)

	state.logger.verbose_json('updated model', model)
	state.logger.stop_step()

	state.convexity_tester.add_model(model)

	# Update plot...

	state.current_plot.add_points(
		model.sample, **state.params.plot_options['sample-points'])
	state.current_plot.add_contour(
		model.unshifted_objective.evaluate,
		**state.params.plot_options['objective-contour']
	)
	unshifted_gradient_direction = model.unshifted_gradient.copy()
	grad_norm = np.linalg.norm(unshifted_gradient_direction)
	if grad_norm > 1e-8:
		unshifted_gradient_direction *= model.r / grad_norm
	state.current_plot.add_arrow(
		state.current_iterate, state.current_iterate - unshifted_gradient_direction,
		**state.params.plot_options['negative-gradient'],
		width=0.05 * state.outer_tr_radius
	)

	state.current_plot.add_lines(
		model.unshifted_A, model.unshifted_b,
		**state.params.plot_options['constraint-contour']
	)

	return model


def _model_accuracy(model, sample, values):
	for point, expected_value in zip(sample, values):
		actual_value = model.evaluate(point)
		agreement = np.abs(actual_value - expected_value) / max(1.0, np.abs(actual_value)) < 1e-4
		make_assertion(agreement,
			'model polynomial does not agree on sample, expected: ' +
			str(expected_value) + ', found: ' + str(actual_value))


def ensure_model_accuracy(state, model, evaluations, f_poly, c_polys):
	points = np.array([e.x for e in evaluations])

	_model_accuracy(model.unshifted_objective, points, [e.objective for e in evaluations])
	for idx, poly in enumerate(model.unshifted_constraints):
		_model_accuracy(poly, points, [e.constraints[idx] for e in evaluations])

	inner_shifted_points = state.sample_region.shift_sample(points)
	_model_accuracy(f_poly, inner_shifted_points, [e.objective for e in evaluations])
	for idx, poly in enumerate(c_polys):
		_model_accuracy(poly, inner_shifted_points, [e.constraints[idx] for e in evaluations])

	outer_shifted_points = (points - model.x) / model.r
	_model_accuracy(model.shifted_objective, outer_shifted_points, [e.objective for e in evaluations])
	for idx, poly in enumerate(model.shifted_constraints):
		_model_accuracy(poly, outer_shifted_points, [e.constraints[idx] for e in evaluations])

	for _ in range(100):
		x = state.current_iterate + 2 * np.random.random() - 1
		eval = state.problem.evaluate(x)
		# print('objective: expected = ' + str(model.unshifted_objective.evaluate(x)))
		# print('objective: actual = ' + str(eval.objective))
		# for i in range(state.num_constraints):
		# 	print('constraint ' + str(i) + ': expected = ' + str(model.unshifted_constraints[i].evaluate(x)))
		# 	print('constraint ' + str(i) + ': actual = ' + str(eval.constraints[i]))

	'''
	sr = state.sample_region
	x = np.random.random(2)
	f_poly.transform(sr.linv).multiply_by_constant(sr.r).translate(sr.center).evaluate(x)
	f_poly.translate(-sr.center).multiply_by_constant(1.0 / sr.r).transform(sr.l).evaluate(x)
	f_poly.evaluate(sr.unshift_point(x))
	'''

