import numpy as np

from utils.assertions import make_assertion
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


def binary_search(bool_function, tol, tmin=1.0, tmax=2.0):
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


def plot_scaling(state, model, old_sample_region, old_radius):
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


def scale_sample_region(state, model, factor, old_radius, tol=1e-12):
	dest = state.current_iterate
	r = state.outer_tr_radius
	# if factor is not None:
	# 	state.sample_region = state.sample_region.scale_towards(dest, factor)
	# 	return

	def bool_func(t):
		scaled = state.sample_region.scale_towards(dest, t)
		# TODO: This should be scaled to lie within the buffered region, not just the trust region...
		feasible = scaled.contained_within_tr(dest, r, tol=tol)
		return feasible, scaled

	old_sample_region = state.sample_region
	state.sample_region = binary_search(bool_func, tol=tol)['value']

	plot_scaling(state, model, old_sample_region, old_radius)

