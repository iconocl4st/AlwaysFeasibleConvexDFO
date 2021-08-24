import numpy as np

from pyomo_opt.project import project
from utils.assertions import make_assertion
from utils.bounds import Bounds
from utils.polyhedron import Polyhedron


def plot_criticality(state, model, criticality, p, is_critical):
	# unshifted_projection = model.x + model.r * p

	bounds = Bounds() \
		.extend(model.x) \
		.extend(model.x - model.r) \
		.extend(model.x + model.r) \
		.extend(p) \
		# .extend(model.x - model.unshifted_gradient) \

	plt = state.plotter.create_plot(
		'criticality_check_' + str(state.iteration),
		bounds.expand(),
		'Iteration ' + str(state.iteration) + ', critical=' + str(is_critical) + ', criticality=' + str(criticality),
		subfolder='criticality'
	)

	plt.add_lines(
		model.unshifted_A, model.unshifted_b,
		**state.params.plot_options['constraint-contour']
	)
	plt.add_linf_tr(model.x, model.r, label='trust region')
	plt.add_arrow(model.x, model.x - model.unshifted_gradient,
		width=0.05 * model.r, color='g', label='negative gradient')
	plt.add_arrow(model.x - model.unshifted_gradient, p,
		width=0.05 * model.r, color='k', label='projection')
	plt.add_arrow(p, model.x,
		width=0.05 * model.r, color='r', label='criticality')
	plt.save()


def check_criticality(state, model):
	state.logger.start_step('Computing criticality')
	success, p, _ = project(
		model.x - model.unshifted_gradient,
		model.unshifted_A, model.unshifted_b,
		state.logger
	)

	criticality = np.linalg.norm(p - model.x)

	state.logger.log_matrix('current iterate', model.x)
	state.logger.log_matrix('projected gradient', p)
	state.logger.verbose('distance to projection: ' + str(criticality))

	make_assertion(success, 'unable to compute criticality measure')
	ret = {
		'converged':
			criticality < state.params.threshold_criticality and
			state.outer_tr_radius < state.params.threshold_tr_radius,
		'critical':
			# not state.unbound_radius and
			criticality < state.params.kappa_crit * state.outer_tr_radius ** state.params.p_delta,
		'criticality': criticality,
	}
	state.logger.info_json('criticality check', ret)
	state.logger.stop_step()

	try:
		plot_criticality(state, model, criticality, p, ret['critical'])
	except:
		print('Unable to make criticality plot')
		raise

	return ret
