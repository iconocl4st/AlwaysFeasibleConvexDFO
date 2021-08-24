import numpy as np

from algorithm.sample_regions.conservative import create_conservative_ellipsoid
from algorithm.sample_regions.numerical import construct_maximal_volume_ellipsoid
from algorithm.sample_regions.recovered import repair_sample_region
from algorithm.sample_regions.scaled import scale_sample_region
from algorithm.sample_regions.spherical import construct_maximal_volume_sphere
from utils.ellipsoid import Ellipsoid


def construct_sample_region(state, model, br):
	n = len(state.current_iterate)

	if br.num_active_constraints == 0:
		state.sample_region = Ellipsoid.create(
			np.eye(n),
			state.current_iterate,
			state.outer_tr_radius
		)

	strategy = state.params.sample_region_strategy
	if strategy == 'conservative-ellipsoid':
		if br.beta0 < 1:
			state.sample_region = create_conservative_ellipsoid(state, model, br)
			return

		sr_method = state.params.empty_sr_method
		if sr_method == 'scale':
			scale_sample_region(state, model, None, old_radius=model.r)
		elif sr_method == 'recover':
			repair_sample_region(state)
		else:
			raise Exception('Unknown empty sample region method: ' + str(sr_method))
	elif strategy == 'max-volume':
		state.sample_region = construct_maximal_volume_ellipsoid(state, model, br)
	elif strategy == 'spherical':
		state.sample_region = construct_maximal_volume_sphere(state, model, br)

