import numpy as np

from utils.bounds import Bounds
from utils.convex_hull import get_convex_hull
from utils.ellipsoid import Ellipsoid
from pyomo_opt.feasible_direction import find_feasible_direction
from algorithm.buffered_region import get_rotation_matrix
from utils.polyhedron import Polyhedron


def _update_largest_feasible(max, t, value):
	if max['t'] is not None and t < max['t']:
		return
	max['t'] = t
	max['value'] = value


def _binary_search(bool_function, tol, tmin=1.0, tmax=2.0):
	maximum = {'t': None, 'value': None}
	while True:
		feasible, value = bool_function(tmax)
		if not feasible:
			break
		_update_largest_feasible(maximum, tmax, value)
		tmin = tmax
		tmax *= 2.0

	while True:
		feasible, value = bool_function(tmin)
		if feasible:
			_update_largest_feasible(maximum, tmin, value)
			break
		tmax = tmin
		tmin /= 2

	while tmax - tmin > tol:
		mid = (tmin + tmax) / 2.0
		feasible, value = bool_function(mid)
		if feasible:
			_update_largest_feasible(maximum, mid, value)
			tmin = mid
		else:
			tmax = mid
	return maximum


def _compute_convex_hull(state):
	n = state.dim
	r = 5
	while True:
		evaluations = state.history.get_evaluations(
			lambda x: np.max(np.abs(state.current_iterate - x)) <= r * state.outer_tr_radius
		)
		if len(evaluations) >= n + 1:
			break
		r += 1
	trial_idx, _ = state.history.find_evaluation(state.current_iterate)
	eindex = [e[0] for e in evaluations]
	locations = np.array([e[1].x for e in evaluations])
	A, b, indices = get_convex_hull(locations)

	faces_containing_iterate = []
	for idx, index_set in enumerate(indices):
		for index in index_set:
			if eindex[index] == trial_idx:
				faces_containing_iterate.append(idx)

	return A, b, locations, faces_containing_iterate


def _compute_feasible_direction(A, fci, locations, iterate):
	# iterate is in the interior...
	if len(fci) == 0:
		u = np.mean(locations, axis=0) - iterate
		return u / np.linalg.norm(u), 1.0

	success, u, theta = find_feasible_direction(A[fci], logger=None)
	assert success, 'implement me'

	if not success:
		u = np.mean(locations, axis=0) - iterate
		return u / np.linalg.norm(u), 1.0
	return u, theta


def repair_sample_region(state):
	n = state.dim

	if len(state.history.evaluations) < n + 1:
		raise Exception("Unable to repair sample region without enough points, is the initial ellipsoid feasible?")

	A, b, locations, fci = _compute_convex_hull(state)
	u, theta = _compute_feasible_direction(A, fci, locations, state.current_iterate)

	# delta = 0.5 * state.outer_tr_radius
	# rot = get_rotation_matrix(u)
	# sample_region = Ellipsoid.create(
	# 	rot.T @ np.diag([1.0] + [theta ** 2 / ((1 - theta) ** 2)] * (n - 1)) @ rot,
	# 	state.current_iterate + delta * u,
	# 	delta / np.sqrt(2)
	# )

	r = state.outer_tr_radius
	poly = Polyhedron(A, b).intersect(
		Polyhedron.create_from_tr(state.current_iterate, r)
	)

	def bool_func(t):
		sphere = Ellipsoid.create(
			np.eye(n),
			state.current_iterate + t * u,
			0.9 * t * theta  # np.sqrt(1 - theta ** 2)
		)
		feasible = sphere.contained_within_polyhedron(poly, tol=1e-12)
		return feasible, sphere

	sample_region = _binary_search(bool_func, tol=1e-12, tmin=0.5 * r, tmax=r)['value']

	plot_repair(state, locations, A, b, fci, sample_region, u)

	state.sample_region = sample_region
	state.outer_tr_radius = np.max(np.abs(sample_region.center - state.current_iterate)) + sample_region.r


def scale_sample_region(state, model, factor, old_radius, tol=1e-12):
	dest = state.current_iterate
	r = state.outer_tr_radius
	# if factor is not None:
	# 	state.sample_region = state.sample_region.scale_towards(dest, factor)
	# 	return

	def bool_func(t):
		scaled = state.sample_region.scale_towards(dest, t)
		feasible = scaled.contained_within_tr(dest, r, tol=tol)
		return feasible, scaled

	old_sample_region = state.sample_region
	state.sample_region = _binary_search(bool_func, tol=tol)['value']

	bounds = Bounds() \
		.extend_tr(state.current_iterate, state.outer_tr_radius) \
		.extend_tr(state.current_iterate, old_radius) \
		.expand()
	plt = state.plotter.create_plot(
		str(state.iteration) + '_scaling_sample_region',
		bounds,
		'scaling sample region',
		subfolder='scaling')
	plt.add_contour(old_sample_region.evaluate, lvls=[-0.1, 0], label='old sample region', color='g')
	plt.add_linf_tr(state.current_iterate, old_radius, label='old trust region', color='g')
	plt.add_contour(state.sample_region.evaluate, lvls=[-0.1, 0], label='new sample region', color='r')
	plt.add_linf_tr(state.current_iterate, state.outer_tr_radius, label='new trust region', color='r')
	if model is not None:
		plt.add_polyhedron(model.get_unshifted_model_constraints(), label='linearized constraints')
	plt.add_point(state.current_iterate, label='iterate')
	plt.save()


def construct_sample_region(state, model, br):
	n = len(state.current_iterate)

	if br.num_active_constraints == 0:
		state.sample_region = Ellipsoid.create(
			np.eye(n),
			state.current_iterate,
			state.outer_tr_radius
		)
	elif br.beta0 < 1:
		gamma = 1 + 2 ** -0.5
		beta = max(0.5, br.beta0)
		state.sample_region = Ellipsoid.create(
			br.rot.T @ np.diag([1.0] + [beta ** 2 / (1 - beta ** 2)] * (n-1)) @ br.rot,
			state.current_iterate + state.outer_tr_radius * br.u / (2 * gamma),
			state.outer_tr_radius / (2 * gamma * np.sqrt(2))
		)
	else:
		scale_sample_region(state, model, None, old_radius=model.r)


def plot_repair(state, locations, A, b, fci, sample_region, u):
	bounds = Bounds()
	for location in locations:
		bounds.extend(location)
	bounds = bounds.expand()
	plt = state.plotter.create_plot(
		'repair_trust_region_' + str(state.iteration),
		Bounds.create(bounds.lb, bounds.ub),
		'restoring sample region', subfolder='repair')
	num_faces = A.shape[0]
	for i in range(num_faces):
		color = 'red' if i in fci else 'blue'
		plt.add_line(A[i], b[i], 'face_' + str(i), color=color)
	plt.add_points(locations, label='nearby evaluations', color='m', marker='x', s=50)
	plt.add_arrow(
		state.current_iterate,
		state.current_iterate + 0.5 * bounds.radius() * u,
		width=state.outer_tr_radius * 0.05,
		label='feasible direction',
		color='green'
	)
	plt.add_point(sample_region.center, label='tr center', color='k', marker='o', s=50)
	plt.add_contour(sample_region.evaluate, color='k', label='repaired sample region', lvls=[0])
	plt.add_linf_tr(
		state.current_iterate, state.outer_tr_radius,
		color='c', label='repaired sample region')
	plt.save()
