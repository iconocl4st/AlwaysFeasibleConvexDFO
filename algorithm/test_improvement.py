import numpy as np

from algorithm.iteration_result import IterationResult

# print('expected decrease: ', expected_decrease)
# print('linearized decrease: ', projected_gradient_decrease)
# print('distance to iteration: ', distance_to_trial)
# print('trust region radius: ', state.outer_tr_radius)
from utils.assertions import make_assertion


def test_for_improvement(state, model, tr_solution):
	state.logger.start_step('testing for improvement')

	current_value = model.shifted_objective.evaluate(model.shifted_x)
	expected_decrease = current_value - tr_solution['trial-value']
	# make_assertion(expected_decrease > 1e-12, 'did not expect decrease: ' + str(expected_decrease))

	distance_to_trial = np.linalg.norm(tr_solution['trial-point'])
	if tr_solution['gc-value'] is not None:
		projected_gradient_decrease = current_value - tr_solution['gc-value']
		make_assertion(projected_gradient_decrease >= -1e-8, 'generalized cauchy did not find decrease')
	else:
		projected_gradient_decrease = None

	state.logger.info_json('checking if decrease expected', {
		'expected-decrease': expected_decrease,
		'cauchy-decrease': projected_gradient_decrease,
		'distance-to-trial': distance_to_trial,
		'trust-region-radius': state.outer_tr_radius,
	})
	if distance_to_trial < 0.1 and (
			projected_gradient_decrease is None or
			expected_decrease < 0.1 * projected_gradient_decrease or
			expected_decrease < 1e-12):
		state.logger.verbose('not enough expected reduction')
		state.logger.stop_step()
		return IterationResult \
			.create('little-expected-reduction') \
			.multiply_radius(state.params.tr_update_dec)

	unshifted_trial_point = model.x + model.r * tr_solution['trial-point']
	evaluation = state.evaluate(unshifted_trial_point)
	if not evaluation.success:
		state.logger.info('infeasible trial point')
		state.logger.stop_step()
		return IterationResult \
			.create('infeasible-trial-point') \
			.multiply_radius(state.params.tr_update_dec)
	actual_value = state.history.find_evaluation(state.current_iterate)[1].objective
	make_assertion(np.abs((current_value - actual_value)) / max(abs(current_value), 1) < 1e-4, \
		'model does not agree with function on current iterate,' \
		+ ' expected: ' + str(current_value) + ', actual: ' + str(actual_value))

	actual_decrease = actual_value - evaluation.objective
	rho = actual_decrease / expected_decrease
	tr_active = np.max(np.abs(tr_solution['trial-point'] - state.current_iterate)) > 0.95 * state.outer_tr_radius

	state.logger.verbose('expected decrease: ' + str(expected_decrease))
	state.logger.verbose('actual decrease: ' + str(actual_decrease))
	state.logger.info('computed rho: ' + str(rho))
	state.logger.verbose('trial at boundary: ' + str(tr_active))
	state.logger.stop_step()

	if rho < state.params.threshold_reduction_minimum:
		return IterationResult \
			.create('insufficient-reduction') \
			.multiply_radius(state.params.tr_update_dec)
	elif rho < state.params.threshold_reduction_minimum:
		return IterationResult \
			.create('minimal-reduction') \
			.multiply_radius(state.params.tr_update_dec) \
			.accept(unshifted_trial_point) \
			.set_successful()
	elif tr_active:
		return IterationResult \
			.create('active-trust-region') \
			.multiply_radius(state.params.tr_update_inc) \
			.accept(unshifted_trial_point) \
			.set_successful()
	else:
		return IterationResult \
			.create('sufficient-reduction') \
			.accept(unshifted_trial_point) \
			.set_successful()
