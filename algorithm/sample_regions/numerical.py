
import numpy as np

from algorithm.sample_regions.spherical import construct_maximal_volume_sphere
from utils.assertions import make_assertion
from utils.bounds import Bounds
from utils.cone_contains_sphere import cone_contains_ellipsoid
from utils.ellipsoid import Ellipsoid
import scipy.stats



# v = nv * vh, |vh| = 1
# cone:
# {x | (x - vertex) @ direction >= b * norm(x - vertex)}

# ellipsoid:
# {x | (x - c) @ q @ (x - c) <= 1}
from utils.stochastic_search import StochasticSearchParams, multi_eval_stochastic_search


def again(vertex, direction, b, q, c):
	ssp = StochasticSearchParams()
	ssp.x0 = c
	ssp.objective = lambda x: (
			(x - c) @ q @ (x - c) - 1 <= 0,
			(x - vertex) @ direction - b * np.linalg.norm(x - vertex)
	)
	def multi_eval(x):
		obj = (x - vertex) @ direction - b * np.linalg.norm(x - vertex, axis=1)
		infeas = np.sum((x - c) * ((x - c) @ q), axis=1) > 1.0
		obj[infeas] = np.nan
		return obj

	ssp.multi_eval = multi_eval
	ssp.initial_radius = 1.0
	ssp.required_objective = 0

	trial_value = multi_eval_stochastic_search(ssp)
	return trial_value.value > 0


class FeasibleRegion:
	def __init__(self):
		self.tr_center = None
		self.tr_radius = None
		self.vhs = None
		self.nvs = None
		self.b = None
		self.pltr = None

	def add_to_plot(self, plot):
		plot.add_point(self.tr_center, label='tr center', color='r', marker='x')
		for nv, vh in zip(self.nvs, self.vhs):
			plot.add_wedge(nv * vh, -vh, self.b, 20.0, color=(1.0, 0.8, 0.6, 0.5), label='cone')
			plot.add_point(nv * vh, label='vertex', color='g')
			plot.add_arrow(nv * vh, nv * vh - vh, label='open direction', color='g')
		plot.add_linf_tr(self.tr_center, self.tr_radius, 'trust region', color='b')

	def contains_ellipsoid(self, q, c):
		for nv, vh in zip(self.nvs, self.vhs):
			m1 = cone_contains_ellipsoid(nv, vh, self.b, q, c)[0]
			m2 = again(nv * vh, -vh, self.b, q, c)
			if m1 != m2:
				print('here')
			# if not m1:
			# 	return False
			if not m2:
				return False
		return True


def determine_ellipsoid(feasible_region, init_center, tol, initial_radius=1.0):
	n = len(init_center)
	init_diag = initial_radius * np.ones_like(init_center)
	init_rot = np.eye(n)
	while not feasible_region.contains_ellipsoid(np.diag(init_diag), init_center):
		init_diag *= 2
		make_assertion(init_diag[0] < 1e30, 'ellipsoid search started at non-interior point')
		# if init_diag[0] > 1e30:
		# 	return -1, init_center, init_rot, None
	init_volume = np.prod(1/init_diag)

	max_volume = None
	max_diag = None
	max_center = None
	max_rot = None
	max_ellipsoid = None

	for i in range(5):
		# print(i)
		volume = init_volume.copy()
		center = init_center.copy()
		rot = init_rot
		diag = init_diag
		ellipsoid = Ellipsoid.create(init_rot.T @ np.diag(init_diag) @ init_rot, init_center, 1.0)

		delta = 1.0
		while delta > 1e-6:
			iterations_since_improvement = 0
			while iterations_since_improvement < 50:
				iterations_since_improvement += 1
				alt_diag = diag + np.random.normal(loc=0, scale=delta, size=n)
				if min(alt_diag) < tol:
					continue
				alt_volume = np.prod(1/alt_diag)
				if alt_volume < volume:
					continue
				alt_rot = scipy.stats.ortho_group.rvs(n)
				# alt_rot = np.real(scipy.linalg.fractional_matrix_power(alt_rot, delta))
				alt_center = center + np.random.normal(loc=0, scale=2 * delta, size=n)

				alt_q = alt_rot.T @ np.diag(alt_diag) @ alt_rot
				try:
					alt_ellipsoid = Ellipsoid.create(alt_q, alt_center, r=1.0)
				except:
					print('exception')
					raise
				if not alt_ellipsoid.contained_within_tr(
						feasible_region.tr_center, feasible_region.tr_radius, tol):
					continue

				if not feasible_region.contains_ellipsoid(alt_q, alt_center):
					continue
				diag = alt_diag
				center = alt_center
				rot = alt_rot
				volume = alt_volume
				ellipsoid = alt_ellipsoid
				iterations_since_improvement = 0
				if feasible_region.pltr:
					feasible_region.pltr.plot_update(volume, center, rot, diag)

			delta /= 2.0

		if max_volume is None or volume > max_volume:
			max_volume = volume
			max_diag = diag
			max_center = center
			max_rot = rot
			max_ellipsoid = ellipsoid
	return max_ellipsoid, max_volume, max_center, max_rot, max_diag


def construct_maximal_volume_ellipsoid(state, model, br):
	# Start at spherical solution...
	spherical_solution = construct_maximal_volume_sphere(state, model, br)

	mve = FeasibleRegion()
	mve.tr_center = state.current_iterate - model.x
	mve.tr_radius = state.outer_tr_radius
	mve.nvs = np.array([model.r * np.linalg.norm(cone.vertex) for cone in br.cones])
	mve.vhs = np.array([-cone.open_direction for cone in br.cones])
	mve.b = br.bdpb

	ellipsoid = determine_ellipsoid(
		mve,
		spherical_solution.center - model.x,
		1e-8,
		initial_radius=2 / spherical_solution.r ** 2)[0]

	p = state.plotter.create_plot(
		filename='initial_ellipsoid',
		title='initial ellipsoid',
		bounds=Bounds()
			.extend_tr(mve.tr_center, mve.tr_radius)
			.extend(state.current_iterate - model.x)
			.expand(1.2),
		subfolder='testing')
	mve.add_to_plot(p)
	p.add_point(np.zeros_like(model.x), label='old center', s=50, marker='o')
	p.add_contour(
		lambda x: spherical_solution.evaluate(model.x + x),
		label='spherical solution',
		color='b',
		lvls=[-0.1, 0.0]
	)

	p.add_contour(
		lambda x: 2 * np.linalg.norm(x - (spherical_solution.center - model.x)) ** 2 / spherical_solution.r ** 2 - 1.0,
		label='spherical solution',
		color='r',
		lvls=[-0.1, 0.0]
	)

	for nv, vh in zip(mve.nvs, mve.vhs):
		feas, pltr = cone_contains_ellipsoid(
			nv, vh, mve.b,
			2 * np.eye(2) / spherical_solution.r ** 2,
			spherical_solution.center - model.x, tol=1e-8)
		print(feas, pltr)
		if pltr is not None:
			pltr(p)

	p.add_contour(
		lambda x: ellipsoid.evaluate(x),
		label='final ellipsoid',
		color='m',
		lvls=[-0.1, 0.0]
	)

	p.save()



	# unshifted_cones = [
	# 	cone.unshift(model.x, model.r)
	# 	for cone in br.cones]
	# [c.get_distance_to(spherical_solution.center) for c in unshifted_cones]
	# shifted = spherical_solution.center - model.x
	# [c.get_distance_to(shifted) for c in br.cones]
	# [c.decompose(shifted) for c in br.cones]


	# mve = FeasibleRegion()
	# mve.tr_center = state.current_iterate - model.x
	# mve.tr_radius = state.outer_tr_radius
	# mve.vs = model.r * br.ws
	# mve.b = br.bdpb
	# ellipsoid = determine_ellipsoid(mve, spherical_solution.center - model.x, 1e-8)[0]

	# Need to shift back from model.x
	raise Exception('implement me!')
