import numpy as np

from algorithm.buffered_region import get_rotation_matrix
from utils.zeros_search import find_zeros


# minimize |x - p|^2
# st		x^T Q x = 1

# x-p = lam * q * x
# (lam * q - I) x = -p
# x = (I - lam * q)^{-1} p
# f(lam) = x^T Q x - 1 =


def _get_x(q, p, lam):
	return np.linalg.solve(np.eye(len(p)) - lam * q, p)


def _f(q, p, lam):
	x = _get_x(q, p, lam)
	return x @ q @ x - 1


def anti_project_point_onto_ellipsoid(q, p, tol=1e-12):
	if q.shape[0] == 1:
		if q[0, 0] < 0.0:
			return False, None, None
		p1 = np.array([+np.sqrt(1 / q[0, 0])], dtype=np.float64)
		p2 = np.array([-np.sqrt(1 / q[0, 0])], dtype=np.float64)
		d1 = np.linalg.norm(p - p1)
		d2 = np.linalg.norm(p - p2)
		return (True, p2, d2) if d1 < d2 else (True, p1, d1)

	eigl, eigv = np.linalg.eig(q)
	asymptotes = [1 / ei for ei in eigl if abs(ei) > tol]
	zav = [(zero, value, _get_x(q, p, zero)) for zero, value in find_zeros(lambda x: _f(q, p, x), asymptotes)]
	zeros = [z for z, _, _ in zav]
	xs = [x for _, _, x in zav]
	distances = [np.linalg.norm(p - x) for x in xs]
	idx = np.argmax(distances)

	if False:
		t = np.linspace(min(asymptotes) - 10, max(asymptotes) + 10, 1000)
		y = np.array([_f(q, p, ti) for ti in t])
		import matplotlib.pyplot as plt
		plt.plot(t, y, t, np.zeros_like(t))
		plt.vlines(asymptotes, -2, 5, colors='red')
		plt.ylabel('the function')
		plt.ylim([-2, 5])
		plt.vlines(zeros, -2, 5, colors='green')
		plt.show()

	return True, xs[idx], distances[idx]


def create_plotter(m, v, c, rhs, max_x):
	def add_to_plot(plot):
		plot.add_contour(
			lambda x: (x - v) @ m @ (x - v),
			label='intermediate_ellipsoid', color='r', lvls=[0])
		plot.add_line(v, 0, label='hyperplane', color='k')
		if rhs is not None:
			plot.add_contour(lambda x: np.linalg.norm(x) ** 2 - rhs, label='intersection', color='c', lvls=[-0.1, 0])
		if max_x is not None:
			plot.add_point(max_x, label='furthest point on sphere', color='r', marker='o', s=50)
	return add_to_plot

# v = nv * vh, |vh| = 1
# cone:
# {x | -(x - v) @ v >= b * norm(v) * norm(x - v)}
# {x | -(x - nv * vh) @ vh >= b * nv * norm(x - nv * vh)}
# {x | -(x - nv * vh) @ vh >= b * nv * norm(x - nv * vh)}

# ellipsoid:
# {x | (x - c) @ q @ (x - c) <= 1}
def cone_contains_ellipsoid(nv, vh, b, q, c, tol=1e-8):
	if abs(nv) < tol:
		return cone_contains_ellipsoid(1.0, vh, b, q, c + vh, tol)[0], None

	v = nv * vh
	# It cant contain the vertex...
	if (v - c) @ q @ (v - c) < 1 + tol:
		return False, None

	# nv = np.linalg.norm(v)
	# vh = v / nv
	rot = get_rotation_matrix(vh)
	m = np.outer(q @ (v - c), q @ (v - c)) - q * (
			v @ q @ v + c @ q @ c - 2 * c @ q @ v - 1)
	mp = rot @ m @ rot.T
	w11 = mp[0, 0]
	w1 = mp[1:, 0]
	w = mp[1:, 1:]

	wc = nv * np.linalg.solve(w, w1)
	wr = nv * (w1 @ wc - nv * w11)

	success, s, dist = anti_project_point_onto_ellipsoid(w / wr, -wc, tol)
	if not success:
		return False, create_plotter(m, v, c, None, None)
	y = wc + s
	x = rot.T @ np.array([0] + [yi for yi in y], dtype=np.float64)
	rhs = (1 / b ** 2 - 1) * nv ** 2

	d = (x - v) / np.linalg.norm(x - v)
	t1 = -(v @ v) / (v @ d)
	t2 = -(v @ q @ d - c @ q @ d) / (d @ q @ d)
	# x1 = v + t1 * d
	# x2 = v + t2 * d

	# abs((x - c) @ q @ (x - c) - 1) < 1e-4
	# print(d @ m @ d)
	# print(v @ (v + t1 * d))
	# print((x2 - c) @ q @ (x2 - c) - 1)

	return np.sign(t1) == np.sign(t2) and x @ x <= rhs, create_plotter(m, v, c, rhs, x)

