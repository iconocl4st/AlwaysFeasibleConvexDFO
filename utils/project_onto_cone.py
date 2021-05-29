import numpy as np


def project_onto_cone(vertex, direction, theta, point):
	dnorm = np.linalg.norm(direction)
	assert dnorm > 1e-8, 'does not support 0 direction'
	direction = direction / dnorm

	x = point - vertex
	nrm = np.linalg.norm(x)
	if nrm < 1e-12:
		return vertex, 0, True

	if x @ direction >= theta * nrm:
		return point, 0, True

	a = np.sqrt((1 - theta ** 2) / (nrm ** 2 - (x @ direction) ** 2))
	d = a * x + (theta - a * x @ direction) * direction
	t = d @ x

	dist_to_edge = np.linalg.norm(t * d - x)
	if t * d @ direction <= 0 <= theta:
		return vertex, nrm

	return vertex + t * d, dist_to_edge


def project_onto_edge_of_cone(vertex, direction, theta, point):
	dnorm = np.linalg.norm(direction)
	assert dnorm > 1e-8, 'does not support 0 direction'
	direction = direction / dnorm

	x = point - vertex
	nrm = np.linalg.norm(x)
	if nrm < 1e-12:
		return vertex, 0, True

	feasible = x @ direction >= theta * nrm

	denom = nrm ** 2 - (x @ direction) ** 2
	if denom < 1e-12:
		# TODO: Not sure if this works when theta < 0
		if feasible:
			# if np.linalg.norm(x / np.linalg.norm(x) - direction) >= 1e-8:
			# 	print('about to fail assertion...')
			# assert np.linalg.norm(x / np.linalg.norm(x) - direction) < 1e-8
			alpha = 1.0
			while True:
				o, _, _ = project_onto_edge_of_cone(
					vertex, direction, theta,
					vertex + alpha * direction + 2 * np.random.random(len(x)) - 1)
				no = np.linalg.norm(o - vertex)
				if no < 1e-12:
					alpha += 1
					continue
				t = (o - vertex) @ x / ((o - vertex) @ (o - vertex))
				p = vertex + t * (o - vertex)
				return p, np.linalg.norm(point - p), True
		else:
			return vertex, nrm, False

	a = np.sqrt((1 - theta ** 2) / denom)
	d = a * x + (theta - a * x @ direction) * direction
	t = d @ x

	dist_to_edge = np.linalg.norm(t * d - x)
	if t * d @ direction <= 0 <= theta:
		return vertex, nrm, feasible

	return vertex + t * d, dist_to_edge, feasible


def sphere_is_contained_within_cone(vertex, direction, theta, point, radius):
	projection, distance, feasible = project_onto_edge_of_cone(vertex, direction, theta, point)
	return feasible and radius < distance, projection
