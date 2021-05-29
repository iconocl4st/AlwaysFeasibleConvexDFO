

def dot_vector(vector, vars):
	if len(vector) == 0:
		return None
	s = vector[0] * vars[0]
	for i in range(1, len(vector)):
		s += vars[i] * vector[i]
	return s


def frebenious(mat):
	s = 0
	for r in mat:
		for e in r:
			s = s + e * e
	return s


def subtract(m1, m2):
	return [
		[e1 - e2 for e1, e2 in zip(r1, r2)]
		for r1, r2 in zip(m1, m2)
	]


def matrix_to_matrix(vector):
	return [[c for c in r] for r in vector]


def vector_to_matrix(vector):
	return [[v] for v in vector]


def transpose(X):
	return list(map(list, zip(*X)))


def negate(X):
	return [[-e for e in r] for r in X]


def sum(X, Y, inner, i, j):
	s = 0
	for k in range(inner):
		s += X[i][k] * Y[k][j]
	return s


def multiply(X, Y):
	n = len(X)
	m = len(Y[0])
	inner = len(X[0])
	if inner != len(Y):
		raise Exception('bad multiplication')
	return [
		[sum(X, Y, inner, i, j) for j in range(m)]
		for i in range(n)
	]


def quadratic_term(variable, matrix):
	vec = vector_to_matrix(variable)
	mat = matrix_to_matrix(matrix)
	return multiply(transpose(vec), multiply(mat, vec))[0][0]


def determinant(matrix):
	if len(matrix) == 1:
		return matrix[0][0]

	ret = 0
	sgn = 1
	for j in range(len(matrix[0])):
		ret = ret + sgn * matrix[0][j] * determinant(
			[
				[
					matrix[ii][jj]
					for jj in range(len(matrix))
					if jj != j
				]
				for ii in range(1, len(matrix))
			]
		)
		sgn = -sgn

	return ret
