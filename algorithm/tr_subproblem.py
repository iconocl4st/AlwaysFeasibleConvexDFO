import numpy as np

from utils.generalized_cauchy import GeneralizedCauchyParams
from utils.generalized_cauchy import compute_generalized_cauchy_point

from pyomo_opt.buffered_tr_subproblem2 import solve_buffered_tr_subproblem
import utils.stochastic_search as sss
import utils.spherical_descent as scd
from utils.trial_value import TrialValue


class BufferedHotStart:
	def __init__(self):
		self.x = None
		self.s = None
		self.t = None

	def apply(self, pyomo_model, use_st=True):
		n = len(self.x)
		m = len(self.t)

		for i in range(n):
			pyomo_model.x[i].set_value(self.x[i])
		if not use_st:
			return

		for i in range(m):
			pyomo_model.t[i].set_value(self.t[i])
		for i in range(m):
			for j in range(n):
				pyomo_model.s[i, j].set_value(self.s[i, j])

	@staticmethod
	def create(x, br):
		n = len(x)
		hotstart = BufferedHotStart()
		hotstart.x = x
		hotstart.s = np.zeros([br.num_active_constraints, n])
		hotstart.t = np.zeros(br.num_active_constraints)
		for i in range(br.num_active_constraints):
			_, hotstart.s[i], hotstart.t[i] = br.cones[i].decompose(x)
		return hotstart


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


def compute_maximum_scaling(tol, br, ugc):
	tmin = 0.0
	tmax = 1.0

	ret = 0, np.zeros_like(ugc)
	assert br.is_buffered(np.zeros_like(ugc), tol=tol), 'no scaling possible'
	for _ in range(10000):
		# point = (1 - tmid) * x + tmid * ugc
		tmid = (tmax + tmin) / 2.0
		point = tmid * ugc
		if br.is_buffered(point):
			ret = tmid, point
			tmin = tmid
		else:
			tmax = tmid
		if tmax - tmin < tol:
			return ret
	assert False, 'binary search should have concluded'


def get_hot_starts(state, model, br, ugc):
	yield BufferedHotStart.create(model.shifted_x, br)
	if ugc is not None and np.linalg.norm(ugc - model.shifted_x) > 1e-12:
		_, point = compute_maximum_scaling(1e-8, br, ugc)
		yield BufferedHotStart.create(point, br)
		yield BufferedHotStart.create(point / 2.0, br)

	search_params = sss.StochasticSearchParams()
	search_params.x0 = model.shifted_x
	search_params.objective = lambda x: tuple([
		np.max(x) < 1 and np.min(x) > -1 and np.all([
			br.cones[i].decompose(x)[0]
			for i in range(br.num_active_constraints)
		]),
		model.shifted_objective.evaluate(x)])
	x_min, _ = sss.simple_stochastic_search(search_params)
	yield BufferedHotStart.create(x_min, br)

	search_params = scd.SphereSearchParams()
	search_params.x0 = (state.sample_region.center - model.x) / model.r
	search_params.objective = model.shifted_objective.evaluate
	search_params.sphere_is_feasible = lambda x, r: \
		np.max(x + r) < 1 and np.min(x - r) > -1 and np.all([
			br.cones[i].contains_sphere(x, r)
			for i in range(br.num_active_constraints)
		])
	if search_params.sphere_is_feasible(search_params.x0, 1e-12):
		x_min, _ = scd.spherical_descent_search(search_params)
		yield BufferedHotStart.create(x_min, br)


def project_onto_linearization(state, model):
	shifted_poly = model.get_all_shifted_constraints()

	gcp = GeneralizedCauchyParams()
	gcp.radius = model.shifted_r
	gcp.x = model.shifted_x
	gcp.model = model.shifted_objective
	gcp.gradient = model.gradient_of_shifted
	gcp.A = shifted_poly.A
	gcp.b = shifted_poly.b
	gcp.cur_val = gcp.model.evaluate(gcp.x)
	gcp.tol = 1e-12
	gcp.plot_each_iteration = True
	gcp.plotter = state.plotter
	gcp.logger = state.logger

	projection, _, _, _ = compute_generalized_cauchy_point(gcp)
	return {
		'gc': projection,
		'gc-value': model.shifted_objective.evaluate(projection) if projection is not None else None,
	}


def solve_tr_subproblem(state, model, br):
	state.logger.start_step('Solving the trust region subproblem')

	projection = project_onto_linearization(state, model)
	state.logger.verbose_json('computed projected gradient', projection)

	tr_subproblem = TrustRegionSubproblem()
	tr_subproblem.m = br.num_active_constraints
	tr_subproblem.n = model.shifted_A.shape[1]
	tr_subproblem.x = model.shifted_x
	tr_subproblem.radius = model.shifted_r
	tr_subproblem.ws = br.ws[br.active_indices]
	tr_subproblem.unit_gradients = model.shifted_A[br.active_indices]
	assert tr_subproblem.m == 0 or np.max(abs(1 - np.linalg.norm(tr_subproblem.unit_gradients, axis=1))) < 1e-12, 'gradients not normalized'
	tr_subproblem.beta = br.bdpb
	tr_subproblem.objective = model.shifted_objective
	tr_subproblem.tol = 1e-8

	trial_value = TrialValue()
	for hotstart in get_hot_starts(state, model, br, projection['gc']):
		# Pyomo sometimes returns answers slightly worse than provided
		# Thus, we attempt hot start values as well
		trial_value.accept(hotstart.x, model.shifted_objective.evaluate(hotstart.x))

		tr_subproblem.hotstart = hotstart
		success, trial, value = solve_buffered_tr_subproblem(tr_subproblem, state.logger)
		if not success:
			print('unable to compute trial step')
			continue
		# assert success, 'unable to compute trial step'
		if np.max(model.shifted_A @ trial - model.shifted_b) > 1e-4:
			print('trial point is not feasible: ' + str(model.shifted_A @ trial - model.shifted_b))
			solve_buffered_tr_subproblem(tr_subproblem, state.logger)
		assert np.max(model.shifted_A @ trial - model.shifted_b) < 1e-4, 'trial point infeasible'

		state.logger.log_matrix('testing trial point', trial)
		state.logger.verbose('with value: ' + str(value))

		trial_value.accept(trial, value)

	assert trial_value.x is not None, 'No solution found'

	state.logger.log_matrix('minimum trial point', trial_value.x)
	state.logger.verbose('with value: ' + str(trial_value.value))

	state.current_plot.add_point(model.x + model.r * trial_value.x, **state.params.plot_options['trial-point'])
	if projection['gc'] is not None:
		state.current_plot.add_point(model.x + model.r * projection['gc'], **state.params.plot_options['projected-gradient-descent'])

	state.logger.stop_step()
	return {'trial-point': trial_value.x, 'trial-value': trial_value.value, **projection}
