import numpy as np


def get_minimum_so_far(values):
	ret = np.zeros_like(values)
	minimum_so_far = np.inf
	for idx, v in enumerate(values):
		minimum_so_far = min(minimum_so_far, v)
		ret[idx] = minimum_so_far
	return ret


def get_maximum_remaining(values):
	ret = np.zeros_like(values)
	maximum_remaining = -np.inf
	for idx, v in [(idx, v) for idx, v in enumerate(values)][::-1]:
		maximum_remaining = max(maximum_remaining, v)
		ret[idx] = maximum_remaining
	return ret


def get_last_interesting_index(values, eps=0.01):
	mini = min(values)
	maxi = max(values)
	minimums_so_far = get_minimum_so_far(values)
	maximum_remainings = get_maximum_remaining(values)

	for idx, (min_sf, max_re) in enumerate(zip(minimums_so_far, maximum_remainings)):
		if (
				min_sf - mini < eps * (maxi - mini) and
				max_re - mini < eps * (maxi - mini)
		):
			return idx, minimums_so_far, maximum_remainings
	return len(values) - 1, minimums_so_far, maximum_remainings


class PerformancePlotType:
	MAX_REMAINING = 'max_remaining'
	MIN_SO_FAR = 'min_so_far'
	ALL = 'all'
	INTERESTING = 'interesting'

	ALL_TYPES = [
		MAX_REMAINING,
		MIN_SO_FAR,
		ALL,
		INTERESTING
	]


def get_performance(history, performance_type):
	oys, oxs = history.successful_indices()

	last_interesting_index, minimums_so_far, maximum_remainings = get_last_interesting_index(oys)
	if performance_type == PerformancePlotType.MAX_REMAINING:
		xs = oxs
		ys = maximum_remainings
	elif performance_type == PerformancePlotType.MIN_SO_FAR:
		xs = oxs
		ys = minimums_so_far
	elif performance_type == PerformancePlotType.ALL:
		xs = oxs
		ys = oys
	elif performance_type == PerformancePlotType.INTERESTING:
		xs = oxs[:last_interesting_index]
		ys = oys[:last_interesting_index]
	else:
		raise Exception('Unknown type: ' + str(type))
	return xs, ys

