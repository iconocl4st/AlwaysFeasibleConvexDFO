import numpy as np

from algorithm.iteration_result import IterationResult

# print('expected decrease: ', expected_decrease)
# print('linearized decrease: ', projected_gradient_decrease)
# print('distance to iteration: ', distance_to_trial)
# print('trust region radius: ', state.outer_tr_radius)
from utils.assertions import make_assertion
from utils.trial_value import TrialValue


def evaluate_trial_points(state, model, tr_solution):
	trial_value = TrialValue()
	min_eval = None
	for solution in [tr_solution['trial']] + tr_solution['heuristics']:
		unshifted_trial_point = model.x + model.r * solution.trial
		evaluation = state.evaluate(unshifted_trial_point)
		if trial_value.accept(solution.trial, evaluation.objective, evaluation.success):
			min_eval = evaluation
	return min_eval, trial_value


def test_for_improvement(state, model, tr_solution):
	state.logger.start_step('testing for improvement')

	current_value = model.shifted_objective.evaluate(model.shifted_x)
	expected_trial_decrease = current_value - tr_solution['trial'].value

	distance_to_trial = np.linalg.norm(tr_solution['trial'].trial)
	make_assertion(tr_solution['linear'] is not None, 'no linear solution')
	projected_gradient_decrease = current_value - tr_solution['linear'].value
	make_assertion(projected_gradient_decrease >= -1e-8, 'generalized cauchy did not find decrease')

	state.logger.info_json('checking if decrease expected', {
		'expected-decrease': expected_trial_decrease,
		'cauchy-decrease': projected_gradient_decrease,
		'distance-to-trial': distance_to_trial,
		'trust-region-radius': state.outer_tr_radius,
	})
	if distance_to_trial < 0.1 and (
			projected_gradient_decrease is None or
			expected_trial_decrease < 0.1 * projected_gradient_decrease or
			expected_trial_decrease < 1e-12):
		state.logger.verbose('not enough expected reduction')
		state.logger.stop_step()
		return IterationResult \
			.create('little-expected-reduction') \
			.multiply_radius(state.params.tr_update_dec)

	evaluation, trial_value = evaluate_trial_points(state, model, tr_solution)
	if evaluation is None:
		state.logger.info('infeasible trial point')
		state.logger.stop_step()
		return IterationResult \
			.create('infeasible-trial-point') \
			.multiply_radius(state.params.tr_update_dec)
	actual_value = state.history.find_evaluation(state.current_iterate)[1].objective
	make_assertion(np.abs((current_value - actual_value)) / max(abs(current_value), 1) < 1e-4, \
		'model does not agree with function on current iterate,' + \
		' expected: ' + str(current_value) + ', actual: ' + str(actual_value))

	feasible_trial_expected_value = model.shifted_objective.evaluate(trial_value.trial)
	feasible_expected_decrease = current_value - feasible_trial_expected_value

	actual_decrease = actual_value - evaluation.objective
	rho = actual_decrease / feasible_expected_decrease
	tr_active = np.max(np.abs(tr_solution['trial'].trial - state.current_iterate)) \
		> 0.95 * state.outer_tr_radius

	state.logger.verbose('expected decrease: ' + str(feasible_expected_decrease))
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
			.accept(evaluation.x) \
			.set_successful()
	elif tr_active:
		return IterationResult \
			.create('active-trust-region') \
			.multiply_radius(state.params.tr_update_inc) \
			.accept(evaluation.x) \
			.set_successful()
	else:
		return IterationResult \
			.create('sufficient-reduction') \
			.accept(evaluation.x) \
			.set_successful()
