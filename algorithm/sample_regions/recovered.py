import numpy as np

from utils.assertions import make_assertion
from utils.bounds import Bounds
from utils.convex_hull import get_convex_hull
from utils.ellipsoid import Ellipsoid
from pyomo_opt.feasible_direction import find_feasible_direction
from utils.polyhedron import Polyhedron

from algorithm.sample_regions.scaled import binary_search


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
	make_assertion(success, 'implement me')

	if not success:
		u = np.mean(locations, axis=0) - iterate
		return u / np.linalg.norm(u), 1.0
	return u, theta


# TODO: This should have implemented a maximum new radius
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

	sample_region = binary_search(bool_func, tol=1e-12, tmin=0.5 * r, tmax=r)['value']

	plot_repair(state, locations, A, b, fci, sample_region, u)

	state.sample_region = sample_region
	state.outer_tr_radius = np.max(np.abs(sample_region.center - state.current_iterate)) + sample_region.r

