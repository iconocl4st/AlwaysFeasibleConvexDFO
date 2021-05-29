import itertools
import scipy.linalg
import numpy as np

from utils.polyhedron import Polyhedron


def _get_hyperplane(vertices):
	n = scipy.linalg.null_space(np.bmat([
		vertices,
		-np.asmatrix(np.ones(vertices.shape[0])).T
	]))
	A_h = n[:-1, :]
	b_h = n[len(n)-1, :]
	nrm = np.linalg.norm(A_h)

	# TODO: Double check this...
	if len(A_h.shape) != 2 or A_h.shape[1] != 1 or len(b_h.shape) != 1:
		return False, None, None

	return True, np.asarray(A_h / nrm).flatten(), b_h[0] / nrm


def _is_valid_hyperplane(vertices, A_h, b_h):
	for vertex in vertices:
		if np.dot(A_h, vertex) > b_h + 1e-10:
			return False
	return True


def get_convex_hull(vertices):
	dim = vertices.shape[1]
	As = []
	bs = []
	vertex_indices = []
	for indices in itertools.combinations(range(len(vertices)), dim):
		success, A_h, b_h = _get_hyperplane(vertices[indices, :])
		if not success:
			continue
		if _is_valid_hyperplane(vertices, +A_h, +b_h):
			As.append(A_h)
			bs.append(b_h)
			vertex_indices.append(indices)
		elif _is_valid_hyperplane(vertices, -A_h, -b_h):
			As.append(-A_h)
			bs.append(-b_h)
			vertex_indices.append(indices)
	return np.array(As), np.array(bs), np.array(vertex_indices)
