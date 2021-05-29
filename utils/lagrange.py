
import numpy as np

from utils.bounds import Bounds
from utils.tr import solve_tr_subproblem
from utils.polynomial import Polynomial
from utils.polynomial import MultiIndex


class LagrangeParams:
	def __init__(self):
		self.basis = None
		self.xsi_replace = None
		self.sample_region = None
		self.plot_creator = None
		self.evaluations = None
		self.plot_maximizations = None
		self.logger = None
		self.plotter = None

	def to_json(self):
		return {
			'basis': self.basis,
			'xsi-replace': self.xsi_replace,
			'sample-region': self.sample_region,
			'evaluations': self.evaluations,
			'plot-maximizations': self.plot_maximizations,
		}


class Certification:
	def __init__(self):
		self.params = None
		self.success = None

		self.num_points = None
		self.basis_dimension = None

		self.values = None
		self.polys = None

		self.current_sample = None
		self.sample_indices = None

		self.Lambda = None
		self.quadratics = None

		# for plotting/debugging
		self.original_sample = None

	def to_json(self):
		return {
			'params': self.params,
			'success': self.success,
			'num-points': self.num_points,
			'values': self.values,
			'polys': self.polys,
			'current-sample': self.current_sample,
			'indices': self.sample_indices,
			'lambda': self.Lambda,
			'quadratics': self.quadratics,
			'original-sample': self.original_sample,
		}

	def get_shifted_points(self):
		return self.current_sample

	def get_lambdas(self):
		return self.polys.copy()

	def shifted_info(self):
		for idx in range(self.basis_dimension):
			yield (
				self.sample_indices[idx],
				self.current_sample[idx],
				self.quadratics[idx]
			)

	def add_to_plot(self, p):
		# self.params.feasibility_enforcer.add_to_plot(p)
		p.add_points(self.original_sample.sample_points, color='r', marker='x', label='original points')
		p.add_points(self.params.trust_region.unshift_points(self.current_sample.sample_points[:self.num_points]), color='b', marker='o', s=50, label='current points')

	@staticmethod
	def create(params):
		c = Certification()
		c.params = params

		c.original_sample = np.array([evaluation[1].x for evaluation in params.evaluations])
		c.sample_indices = [evaluation[0] for evaluation in params.evaluations]
		n_missing_evals = len(params.basis) - c.original_sample.shape[0]
		if n_missing_evals > 0:
			c.original_sample = np.vstack([
				c.original_sample,
				np.repeat(c.original_sample[0][np.newaxis, :], axis=0, repeats=n_missing_evals)
			])
			c.sample_indices += [c.sample_indices[0]] * n_missing_evals
		c.num_points = c.original_sample.shape[0]
		c.basis_dimension = len(params.basis)
		c.current_sample = params.sample_region.shift_sample(c.original_sample)
		c.values = MultiIndex.construct_vandermonde(params.basis, c.current_sample)
		c.polys = np.eye(c.basis_dimension)
		c.success = True
		return c


def _test_v(c):
	V = MultiIndex.construct_vandermonde(c.params.basis, c.get_shifted_points())
	lambdas = c.polys
	reduced = c.values
	ratio = np.linalg.norm(V@lambdas.T - reduced) / np.linalg.norm(reduced)
	if ratio > 1e-10:
		print("TODO: The error here should not be so big...")
	assert ratio < 1e-2, 'Failure to maintain structure of V: ' + str(ratio)


def _find_replacement(c, coef, params, name):
	poly = Polynomial.construct_polynomial(params.basis, coef)
	quad = poly.to_matrix_form()
	success, ppoint, _ = solve_tr_subproblem(quad.g, 2 * quad.Q)
	assert success, 'Unable to optimize lagrange polynomial'

	success, npoint, _ = solve_tr_subproblem(-quad.g, -2 * quad.Q)
	assert success, 'Unable to optimize lagrange negative polynomial'

	pval = quad.evaluate(ppoint)
	nval = quad.evaluate(npoint)

	if params.plot_maximizations:
		p = params.plotter.create_plot(
			'lagrange_maximization_' + name,
			Bounds.create([-1.2, -1.2], [1.2, 1.2]),
			str(poly),
			subfolder='lagrange')
		p.add_point(npoint, label='negative minimizer', s=50, color='red')
		p.add_point(ppoint, label='positive minimizer', s=50, color='green')
		p.add_contour(quad.evaluate, label='polynomial', color='blue')
		p.add_circle(center=np.zeros_like(ppoint), radius=1, label='trust region')
		p.save()

	if abs(nval) > abs(pval):
		return npoint, nval
	else:
		return ppoint, pval

	# params.logger.log_message("Maximum absolute value of " + str(poly) + " over tr is " + str(mval))
	# return mpoint, mval


def _swap_rows(c, i1, i2):
	if i1 == i2:
		return

	t = c.sample_indices[i1]
	c.sample_indices[i1] = c.sample_indices[i2]
	c.sample_indices[i2] = t

	t = c.values[i1].copy()
	c.values[i1] = c.values[i2]
	c.values[i2] = t

	t = c.current_sample[i1].copy()
	c.current_sample[i1] = c.current_sample[i2]
	c.current_sample[i2] = t


def _replace_point(c, idx, shifted_point):
	c.values[idx, :] = (
		np.array([b.as_exponent(shifted_point) for b in c.params.basis]) @ c.polys.T
	)
	c.current_sample[idx] = shifted_point
	c.sample_indices[idx] = -1


def _reduce(c, i):
	pivot = c.values[i, i]

	c.values[:, i] /= pivot
	c.polys[i, :] /= pivot

	for j in range(c.basis_dimension):
		if i == j:
			continue
		coef = c.values[i, j]
		c.values[:, j] -= coef * c.values[:, i]
		c.polys[j, :] -= coef * c.polys[i, :]


def perform_lu_factorization(params):
	params.logger.start_step('Computing lagrange polynomials')

	c = Certification.create(params)
	_test_v(c)

	for i in range(c.basis_dimension):
		_test_v(c)

		params.logger.log_matrix('LU Factorization, step ' + str(i) + ", values", c.values)
		params.logger.log_matrix('LU Factorization, step ' + str(i) + ", polynomials", c.polys)

		# do not replace first point
		if i == 0:
			pivot_index = 0
		else:
			pivot_index = i + np.argmax(np.abs(c.values[i:, i]))
		pivot_value = abs(c.values[pivot_index, i])
		if pivot_value < params.xsi_replace:
			point, val = _find_replacement(c, c.polys[i, :].copy(), params, '')
			assert not np.isnan(val) and abs(val) > params.xsi_replace, 'Unable to find replacement point' + str(abs(val)) + str(params.xsi_replace)
			_replace_point(c, i, point)
			_test_v(c)
			params.logger.verbose("Replaced point at " + str(i) + " with " + str([xi for xi in point]))
		else:
			_swap_rows(c, i, pivot_index)
			_test_v(c)

		params.logger.verbose("Row " + str(i) + ", pivot value of " + str(pivot_value))
		_reduce(c, i)
		_test_v(c)

	_test_v(c)

	# TODO: This might not include all points...
	n = c.current_sample.shape[1]
	p = params.plotter.create_plot(
		'model_improvement',
		Bounds.create(-2.2 * np.ones(n), 2.2 * np.ones(n)),
		'sample set improvement',
		subfolder='lagrange')

	original = params.sample_region.shift_sample(c.original_sample)
	improved = c.current_sample
	has = lambda arr, x: any([np.linalg.norm(x - v) < 1e-12 for v in arr])

	discarded = np.array([x for x in original if not has(improved, x)])
	kept = np.array([x for x in original if has(improved, x)])
	added = np.array([x for x in improved if not has(original, x)])
	if len(discarded) > 0:
		p.add_points(discarded, label='discarded points', color='red', marker='x')
	if len(kept) > 0:
		p.add_points(kept, label='kept points', color='blue', marker='+', s=50)
	if len(added) > 0:
		p.add_points(added, label='added points', color='green',  marker='o')
	p.add_circle(np.zeros(2), 1, label='trust region')
	p.save()

	c.quadratics = []
	for i in range(c.polys.shape[0]):
		poly = Polynomial.construct_polynomial(params.basis, c.polys[i, :])
		quad = poly.to_matrix_form()
		c.quadratics.append(quad)

		success, ppoint, pval = solve_tr_subproblem(quad.g, 2 * quad.Q)
		assert success, 'Unable to compute lambda'
		success, npoint, nval = solve_tr_subproblem(-quad.g, -2 * quad.Q)
		assert success, 'Unable to compute lambda'
		Lambda = max(abs(pval), abs(nval))
		if c.Lambda is None or Lambda > c.Lambda:
			c.Lambda = Lambda

		for idx in range(c.basis_dimension):
			x = c.current_sample[idx]
			val = poly.evaluate(x)
			if i == idx:
				assert np.abs(1.0 - val) < 1e-8, 'lagrange polynomial failed to be one at its own point' \
					', found: ' + str(val)
			else:
				assert np.abs(val) < 1e-8, 'lagrange polynomial failed to be zero at another point' \
					', found: ' + str(val)

			val = quad.evaluate(x)
			if i == idx:
				assert np.abs(1.0 - val) < 1e-8, 'lagrange polynomial failed to be one at its own point' \
					', found: ' + str(val)
			else:
				assert np.abs(val) < 1e-8, 'lagrange polynomial failed to be zero at another point' \
					', found: ' + str(val)

		if params.plot_maximizations:
			name = str(poly) + ', lambda = ' + str(Lambda)
			p = params.plotter.create_plot(
				'lambda_calculation',
				Bounds.create([-1.2, -1.2], [1.2, 1.2]),
				name,
				subfolder='lagrange')
			p.add_contour(quad.evaluate, label='lagrange polynomial')
			p.add_points(c.current_sample, label='replacement points', color='red', marker='x')
			p.add_point(ppoint, label='min value = ' + str(pval), color='green', marker='o')
			p.add_point(npoint, label='max value = ' + str(nval), color='blue', marker='o')
			p.add_circle(np.zeros(2), 1, label='trust region', color='k')
			p.save()

	params.logger.verbose('Lambda = ' + str(c.Lambda))
	for idx, quad in enumerate(c.quadratics):
		params.logger.verbose_json('lagrange polynomial ' + str(idx), quad)
	params.logger.stop_step()
	return c
