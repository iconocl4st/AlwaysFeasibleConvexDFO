import numpy as np
from algorithm.sample_regions.scaled import scale_sample_region
from utils.ellipsoid import Ellipsoid


def create_conservative_ellipsoid(state, model, br):
	n = model.n
	gamma = 1 + 2 ** -0.5
	beta = max(0.5, br.beta0)
	return Ellipsoid.create(
		q=br.rot.T @ np.diag([1.0] + [beta ** 2 / (1 - beta ** 2)] * (n - 1)) @ br.rot,
		center=state.current_iterate + state.outer_tr_radius * br.u / (2 * gamma),
		r=state.outer_tr_radius / (2 * gamma * np.sqrt(2))
	)
