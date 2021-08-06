import os
import json

import numpy as np

from pyomo_opt.project import project
from settings import EnvironmentSettings
from utils.assertions import make_assertion
from utils.bounds import Bounds
from utils.json_utils import JsonUtils
from utils.plotting import Plotting
from utils.quadratic import Quadratic


class GeneralizedCauchyParams:
	def __init__(self):
		self.radius = None
		self.x = None
		self.gradient = None
		self.model = None
		self.A = None
		self.b = None
		self.cur_val = None
		self.plot_each_iteration = False
		self.plotter = None

		self.k_lbs = 0.75
		self.k_ubs = 0.25
		self.k_trd = 0.50
		self.k_epp = 0.25

		self.tol = 1e-4
		self.log_failure = True

	def to_json(self):
		return {
			'radius': self.radius,
			'x': self.x,
			'gradient': self.gradient,
			'model': self.model,
			'A': self.A,
			'b': self.b,
			'current-value': self.cur_val,
			'kappa_lbs': self.k_lbs,
			'kappa_ubs': self.k_ubs,
			'kappa_trd': self.k_trd,
			'kappa_epps': self.k_epp,
			'tolerance': self.tol
		}

	@staticmethod
	def parse_json(json_object):
		params = GeneralizedCauchyParams()
		params.radius = json_object['radius']
		params.x = np.array(json_object['x'], dtype=np.float64)
		params.gradient = np.array(json_object['gradient'], dtype=np.float64)
		params.model = Quadratic.parse_json(json_object['model'])
		params.gradient = np.array(json_object['gradient'], dtype=np.float64)
		params.A = np.array(json_object['A'], dtype=np.float64)
		params.b = np.array(json_object['b'], dtype=np.float64)
		params.cur_val = np.array(json_object['current-value'])
		params.k_lbs = json_object['kappa_lbs']
		params.k_ubs = json_object['kappa_ubs']
		params.k_trd = json_object['kappa_trd']
		params.k_epp = json_object['kappa_epps']
		params.tol = json_object['tolerance']
		params.plot_each_iteration = False
		return params


def _create_the_books_plot(gcp):
	import matplotlib.pyplot as pyplt
	pyplt.close()
	N = 100
	tt = np.linspace(0, 0.005, N)
	ss = np.zeros([N, 2])
	for i in range(N):
		ss[i] = gcp.x - tt[i] * gcp.gradient
	yvals = np.zeros(N)
	for i in range(N):
		yvals[i] = gcp.cur_val + gcp.k_ubs * gcp.gradient @ (ss[i] - gcp.x)
	zvals = np.zeros(N)
	for i in range(N):
		zvals[i] = gcp.model.evaluate(ss[i])
	vvals = np.zeros(N)
	for i in range(N):
		vvals[i] = gcp.cur_val + gcp.k_lbs * gcp.gradient @ (ss[i] - gcp.x)

	# uvals = np.zeros(N)
	# for i in range(N):
	#	uvals[i] = gcp.cur_val - tt[i] * gcp.k_ubs * gcp.gradient @ gcp.gradient

	pyplt.plot(tt, yvals, label='upper bound')
	pyplt.plot(tt, zvals, label='function values')
	pyplt.plot(tt, vvals, label='lower bound')
	pyplt.legend()
	pyplt.show()


def _do_projection(x, A, b, tol=1e-12, hotstart=None):
	if np.max(A @ x - b) < tol:
		return x
	success, ret, _ = project(x, A, b, hotstart=hotstart, tol=tol)
	assert success, 'Unable to project negative gradient'
	return ret


def _binary_search(gcp, t):
	r = t / 2.0
	mid_t = t
	lower_t = mid_t - r
	upper_t = mid_t + r
	lower_val = None
	mid_val = None
	upper_val = None
	lower_projected_gradient = None; mid_projected_gradient = None; upper_projected_gradient = None
	while True:
		if lower_val is None:
			lower_scaled_gradient = gcp.x - lower_t * gcp.gradient
			lower_projected_gradient = _do_projection(lower_scaled_gradient, gcp.A, gcp.b, tol=1e-12)
			lower_val = gcp.model.evaluate(lower_projected_gradient)
		if upper_val is None:
			upper_scaled_gradient = gcp.x - upper_t * gcp.gradient
			upper_projected_gradient = _do_projection(upper_scaled_gradient, gcp.A, gcp.b, tol=1e-12)
			upper_val = gcp.model.evaluate(upper_projected_gradient)
		if mid_val is None:
			mid_scaled_gradient = gcp.x - mid_t * gcp.gradient
			mid_projected_gradient = _do_projection(mid_scaled_gradient, gcp.A, gcp.b, tol=1e-12)
			mid_val = gcp.model.evaluate(mid_projected_gradient)


		completed = upper_t - lower_t <= gcp.tol

		if mid_val <= lower_val and mid_val <= upper_val:
			if completed:
				return mid_t, mid_projected_gradient, mid_val
			r /= 2.0
			lower_t = mid_t - r
			upper_t = mid_t + r
			lower_val = None
			upper_val = None
		elif lower_val <= mid_val and lower_val <= upper_val:
			if completed:
				return lower_t, lower_projected_gradient, lower_val
			assert lower_t / 2.0 > 0.0, 'positive binary search with middle at 0'

			r = min(r, lower_t / 2.0)
			mid_t = lower_t
			mid_val = lower_val
			lower_t = mid_t - r
			upper_t = mid_t + r
			lower_val = None
			upper_val = None
		elif upper_val <= lower_val and upper_val <= mid_val:
			if completed:
				return upper_t, upper_projected_gradient, upper_val
			lower_t = mid_t
			mid_t = upper_t
			upper_t = mid_t + r
			lower_val = mid_val
			mid_val = upper_val
			upper_val = None

			far_upper_t = 2 * upper_t
			far_upper_scaled_gradient = gcp.x - far_upper_t * gcp.gradient
			far_upper_projected_gradient = _do_projection(far_upper_scaled_gradient, gcp.A, gcp.b, tol=1e-12)
			far_upper_val = gcp.model.evaluate(far_upper_projected_gradient)
			if far_upper_val < lower_val and far_upper_val < mid_val:
				while far_upper_val < lower_val and far_upper_val < mid_val:
					mid_t = far_upper_t
					mid_val = far_upper_val
					r *= 2
					far_upper_t *= 2

					far_upper_scaled_gradient = gcp.x - far_upper_t * gcp.gradient
					far_upper_projected_gradient = _do_projection(far_upper_scaled_gradient, gcp.A, gcp.b, tol=1e-12)
					far_upper_val = gcp.model.evaluate(far_upper_projected_gradient)
				lower_t = mid_t - r
				upper_t = mid_t + r
				lower_val = None
				upper_val = None
		else:
			assert False, 'one of the points has to be the minimum'


def compute_generalized_cauchy_point(gcp):
	tmin = 0
	# I don't like the following line...
	t = gcp.radius / max(1e-4, min(20.0, np.linalg.norm(gcp.gradient)))
	tmax = None
	s = None

	projections = []
	tangent_cone_projections = []
	ts = [t]

	for k in range(100):
		scaled_gradient = gcp.x - t * gcp.gradient
		s = _do_projection(scaled_gradient, gcp.A, gcp.b, hotstart=s, tol=1e-12)
		projections.append(s)

		active_indices = gcp.A @ s - gcp.b > -gcp.tol

		if gcp.A[active_indices].shape[0] == 0:
			l = -gcp.gradient
		else:
			l = _do_projection(-gcp.gradient, gcp.A[active_indices], np.zeros(sum(active_indices)), tol=1e-12)
		tangent_cone_projections.append(l)

		val = gcp.model.evaluate(s)

		nrm = np.linalg.norm(s - gcp.x)
		dotpr = gcp.gradient @ (s - gcp.x)

		# eqn_12_2_1_a = nrm <= gcp.radius
		# eqn_12_2_1_b = val <= gcp.cur_val + gcp.k_ubs * dotpr
		#
		# eqn_12_2_2_a = nrm >= gcp.k_trd * gcp.radius
		# eqn_12_2_2_b = val >= gcp.cur_val + gcp.k_lbs * dotpr
		# eqn_12_2_2_c = np.linalg.norm(l) <= gcp.k_epp * np.abs(dotpr) / gcp.radius
		#
		# print('t=' + str(t))
		# print(eqn_12_2_1_a)
		# print(eqn_12_2_1_b)
		#
		# print(eqn_12_2_2_a)
		# print(eqn_12_2_2_b)
		# print(eqn_12_2_2_c)

		if nrm > gcp.radius or val > gcp.cur_val + gcp.k_ubs * dotpr:
			tmax = t
		elif nrm < gcp.k_trd * gcp.radius and \
				val < gcp.cur_val + gcp.k_lbs * dotpr and \
				np.linalg.norm(l) > gcp.k_epp * np.abs(dotpr) / gcp.radius:
			tmin = t
		else:
			if False:
				gcp.logger.log_message('beginning binary search on projected gradient...')
				bst, bsx, bsf = _binary_search(gcp, t)
				gcp.logger.log_json('...done:', {
					'tr-methods': {
						'x': s,
						'objective': val,
					},
					'binary-search': {
						'x': bsx,
						'objective': bsf,
					},
				})
			return s, val, projections, tangent_cone_projections

		if tmax is None:
			t *= 2
		else:
			t = (tmin + tmax) / 2.0
		ts.append(t)

		if gcp.plot_each_iteration:
			try:
				bounds = Bounds()
				bounds.extend(gcp.x)
				bounds.extend(gcp.x - gcp.gradient)
				bounds.extend(s)
				bounds = bounds.expand()
				plt = gcp.plotter.create_plot(
					filename='generalized_cauchy_iteration_' + str(k),
					bounds=bounds,
					title='Generalized Cauchy Point',
					subfolder='gc'
				)
				# os.makedirs(os.path.dirname(filename), exist_ok=True)
				plt.add_lines(gcp.A, gcp.b, label='constraint', color='m')
				plt.add_point(gcp.x, label='center', color='green')
				plt.add_point(gcp.x - gcp.gradient, label='negative gradient', color='red')
				plt.add_arrow(
					gcp.x,
					gcp.x - gcp.gradient,
					label='negative gradient', color='c',
					width=0.005 * np.min(bounds.ub - bounds.lb))
				plt.add_point(s, label='projected gradient path', color='blue')
				plt.add_point(gcp.x + l, label='projected tangent', color='yellow')
				plt.save()
			except:
				gcp.logger.log_message('unable to plot generalized cauchy iteration')

	if gcp.log_failure:
		with open(EnvironmentSettings.get_output_path(['gc_failure.json']), 'w') as failure_out:
			JsonUtils.dump({'failure-location': 'generalized cuachy', 'params': gcp}, failure_out)

	# return s, val, projections, tangent_cone_projections
	# make_assertion(False, "The generalized Cauchy point calculation shouldn't take this many iterations")
	return None, None, None, None


if __name__ == '__main__':
	with open(EnvironmentSettings.get_output_path(['gc_failure.json']), 'r') as failure_in:
		gcp = GeneralizedCauchyParams.parse_json(json.load(failure_in)['params'])
		gcp.log_failure = False
		compute_generalized_cauchy_point(gcp)

