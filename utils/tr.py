import numpy as np

'''
Problem:
min		g @ x + 0.5 * x.T @ q @ x
s.t.	x.T @ x <= 1


First order Optimality Conditions:
b + q @ x = mu * x => (q - mu * I) @ x = -b
p(mu) = -np.linalg.solve(q + mu * np.eye(n), b)
'''


class _BinarySearchBounds:
	def __init__(self):
		self.xmin = None
		self.xmax = None


def _binary_search(bounds, f, tol):
	assert bounds.xmin <= bounds.xmax, 'Invalid arguments to binary search'
	assert f(bounds.xmin) >= 0, 'Lower bound not satisfied'
	assert f(bounds.xmax) <= 0, 'Lower bound not satisfied'
	# TODO: Can the following be an infinite loop?
	while True:
		mid = (bounds.xmin + bounds.xmax) / 2.0
		val = f(mid)
		if np.abs((bounds.xmax - bounds.xmin) / bounds.xmax) < tol:
			return mid, val
		if val < 0:
			bounds.xmax = mid
		else:
			bounds.xmin = mid


def _find_upper_bound(f, bounds):
	lower = bounds.xmin
	val = 1.0
	while True:
		try:
			fval = f(lower + val)
		except np.linalg.LinAlgError:
			# Not so sure this can happen anymore
			pass
		else:
			if fval < 0:
				bounds.xmax = lower + val
				return
			elif fval < 1e10:
				bounds.xmin = lower + val
		val *= 2


def _find_finite_lower_bound(f, bounds):
	lower = bounds.xmin
	val = bounds.xmax - bounds.xmin
	while val > 0:
		val /= 2.0
		bounds.xmin = lower + val
		fval = f(bounds.xmin)
		if fval >= 0:
			return
		bounds.xmax = bounds.xmin

	assert False, 'should not be able to end up here...'


def _find_bounds(f, singularity):
	bounds = _BinarySearchBounds()
	bounds.xmin = singularity
	_find_upper_bound(f, bounds)
	_find_finite_lower_bound(f, bounds)
	return bounds


def _ret(x, g, q):
	val = g @ x + 0.5 * x.T @ q @ x
	assert not np.isnan(val)
	return True, x, val


def solve_tr_subproblem(g, q, tol=1e-12):
	n = len(g)

	if np.linalg.norm(q) < tol:
		x = -g
		nrm = np.linalg.norm(x)
		if nrm < tol:
			x = np.zeros(n, dtype=np.float)
			x[0] = 1.0
			return _ret(x, g, q)
		return _ret(x / nrm, g, q)

	lam, v = np.linalg.eig(q)
	assert np.linalg.norm(q @ v - v @ np.diag(lam)) < 1e-4, 'failed to compute eigenvalues'

	min_idx = np.argmin(lam)
	min_lam = lam[min_idx]

	eye = np.eye(n)
	if np.abs(v[:, min_idx] @ g) < tol:
		p = np.zeros_like(g)
		sum_squares = 0
		for i in range(n):
			if np.abs(lam[i] - min_lam) < tol:
				continue
			scalar = v[:, i] @ g / (lam[i] - min_lam)
			p -= scalar * v[:, i]
			sum_squares += scalar ** 2
		if 1 >= sum_squares:
			x = p + np.sqrt(1 - sum_squares) * v[:, min_idx]
			return _ret(x, g, q)
		else:
			print('hit odd case from Nocedal and Wright')

	p = lambda t: np.linalg.norm(np.linalg.solve(q + t * eye, g)) - 1
	if min_lam > tol:
		x = np.linalg.solve(q, -g)
		if np.linalg.norm(x) <= 1:
			return _ret(x, g, q)

	if False:
		import matplotlib.pyplot as plt
		t = np.linspace(-5, 5, 100)
		toplot = np.array([p(ti) for ti in t])
		plt.plot(t, toplot)
		plot.show()

	# TODO: Newton's method would be faster
	bounds = _find_bounds(p, singularity=max(0, -min_lam))
	mu, val = _binary_search(bounds, p, tol=tol)

	x = np.linalg.solve(q + mu * eye, -g)
	assert np.linalg.norm(q @ x + g + mu * x) < 1e-4, 'Problem not solved'
	return _ret(x, g, q)
