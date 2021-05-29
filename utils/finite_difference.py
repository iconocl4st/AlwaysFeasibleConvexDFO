from mimetypes import init

import numpy as np

from algorithm.buffered_region import get_rotation_matrix
from pyomo_opt.feasible_direction import find_feasible_direction
from utils.ellipsoid import Ellipsoid


def construct_gradient(function, initial, delta):
	ei = np.zeros_like(initial, dtype=np.float64)
	grad = np.zeros_like(initial, dtype=np.float64)
	for i in range(len(initial)):
		ei[i] = delta
		pd = function(initial + ei)
		md = function(initial - ei)
		ei[i] = 0

		grad[i] = (pd - md) / (2 * delta)
	return grad


def construct_initial_ellipsoid(constraints, initial, delta=1e-2):
	n = len(initial)
	if len(constraints) == 0:
		return True, initial, np.eye(len(initial)), 1
	A = np.array([
		construct_gradient(constraint_function, initial, delta)
		for constraint_function in constraints
	])
	c = np.array([
		constraint_function(initial)
		for constraint_function in constraints
	])

	grad_norms = np.linalg.norm(A, axis=1)
	if np.max(c) < -1e-2:
		# if the gradient norm is zero and the constraint value is not zero,
		# then the linearization tells us nothing
		idcs = np.logical_or(grad_norms > 1e-4, -c < 1e-4)
		if sum(idcs) == 0:
			return True, initial, np.eye(len(initial)), 1.0
		return True, initial, np.eye(len(initial)), 0.75 * np.min(-c[idcs] / grad_norms[idcs])

	min_rad = np.min(grad_norms[c < -1e-2] / -c[c < -1e-2]) / 2

	success, u, t = find_feasible_direction(gradients=A[c >= -1e-2], logger=None, tol=None)
	if success:
		gamma = 1 + 2 ** -0.5
		# beta = max(0.5, t)
		beta = min(0.5, t)
		rot = get_rotation_matrix(u)
		q = rot.T @ np.diag([1.0] + [beta ** 2 / (1 - beta ** 2)] * (n-1)) @ rot
		center = initial + min_rad * u / (2 * gamma)
		r = min_rad / (2 * gamma * np.sqrt(2))
		return True, center, q, r / 2

	print(A, c)


