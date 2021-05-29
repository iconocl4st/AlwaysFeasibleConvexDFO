import numpy as np

from pyomo_opt.feasible_direction import find_feasible_direction
from utils.bounds import Bounds
from utils.polyhedron import Polyhedron
from utils.project_onto_cone import project_onto_edge_of_cone
from utils.project_onto_cone import sphere_is_contained_within_cone


# class BufferingParaboloid:
# 	def __init__(self):
# 		self.vertex = None
# 		self.open_direction = None
# 		self.open_width = None
#
# 	def decompose(self, x, tol=1e-12):
# 		s = x - self.vertex
# 		nrm = np.linalg.norm(s)
# 		if nrm < tol:
# 			return True, s, 0.0
# 		return s @ self.open_direction >= nrm * self.open_width, s / nrm, nrm
#
# 	def unshift(self, x, r):
# 		return Cone.create(
# 			x + r * self.vertex,
# 			self.open_direction,
# 			self.open_width)
#
# 	def to_pyomo(self, model, i):
# 		model.constraints.add(
# 			-sum(model.s[i, j] * self.open_direction[i, j]
# 			for j in range(n)) >= self.open_width)
#
# 	def add_to_plot(self, plot_obj, radius):
# 		plot_obj.add_wedge(self.vertex, self.open_direction, self.open_width, radius=radius, color='yellow')
#
# 	def to_json(self):
# 		return {'vertex': self.vertex, 'direction': self.open_direction, 'width': self.open_width}
#
# 	@staticmethod
# 	def create(vertex, open_direction, beta):
# 		cone = Cone()
# 		cone.vertex = vertex
# 		cone.open_direction = open_direction
# 		nrm = np.linalg.norm(open_direction)
# 		assert 1e-12 < nrm < 1e100, 'implement me: how to normalize?'
# 		cone.open_direction = open_direction / nrm
# 		cone.open_width = beta
# 		return cone


class Cone:
	def __init__(self):
		self.vertex = None
		self.open_direction = None
		self.open_width = None

	def contains_sphere(self, center, radius):
		return sphere_is_contained_within_cone(
			self.vertex, self.open_direction, self.open_width, center, radius)[0]

	def decompose(self, x, tol=1e-12):
		s = x - self.vertex
		nrm = np.linalg.norm(s)
		if nrm < tol:
			return True, s, 0.0
		return s @ self.open_direction >= nrm * self.open_width, s / nrm, nrm

	def unshift(self, x, r):
		return Cone.create(
			x + r * self.vertex,
			self.open_direction,
			self.open_width)

	def to_pyomo(self, model, i):
		model.constraints.add(
			-sum(model.s[i, j] * self.open_direction[i, j] for j in range(n)) >= self.open_width)

	def add_to_plot(self, plot_obj, radius):
		plot_obj.add_wedge(self.vertex, self.open_direction, self.open_width, radius=radius,
			color='yellow', label='buffering cone')

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		return '{x=' + str(self.vertex) + '+s|s@' + str(self.open_direction) + '>=' + str(self.open_width) + '|s|}'

	def to_json(self):
		return {'vertex': self.vertex, 'direction': self.open_direction, 'width': self.open_width}

	@staticmethod
	def create(vertex, open_direction, beta):
		cone = Cone()
		cone.vertex = vertex
		cone.open_direction = open_direction
		nrm = np.linalg.norm(open_direction)
		assert 1e-12 < nrm < 1e100, 'implement me: how to normalize?'
		cone.open_direction = open_direction / nrm
		cone.open_width = beta
		return cone


class BufferedRegion:
	def __init__(self):
		self.active_indices = None
		self.num_active_constraints = None
		self.zs = None
		self.ws = None
		self.u = None
		self.pi = None
		self.bdpb = None
		self.adpa = None
		self.beta0 = None
		self.rot = None
		self.cones = None
		self.regular = None

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		return 'buffered region'

	def is_buffered(self, x, tol=1e-12):
		for cone in self.cones:
			if not cone.decompose(x, tol=tol)[0]:
				return False
		return True

	def to_json(self):
		return {
			'active-constraints': self.active_indices,
			'zs': self.zs,
			'ws': self.ws,
			'u': self.u,
			'pi': self.pi,
			'beta * Delta ** p_beta': self.bdpb,
			'1 - alpha * Delta ** p_alpha': self.adpa,
			'beta0': self.beta0,
			'rot':  self.rot,
			'cones': self.cones,
		}


def get_rotation_matrix(u):
	e1 = np.zeros(len(u))
	e1[0] = 1.0
	e1u = e1 + u
	rot = 2 * np.outer(e1u, e1u) / np.inner(e1u, e1u) - np.eye(len(u))
	return rot


def create_buffering_plot(state, model, br):
	buffering_plot = state.plotter.create_plot(
		'buffering_cones_' + str(state.iteration),
		Bounds.create(
			model.shifted_x - 1.2 * (1 + state.params.ka),
			model.shifted_x + 1.2 * (1 + state.params.ka)),
		'Buffered Region',
		subfolder='buffered_regions'
	)

	# TODO: move to where we can also plot the next sample region...
	state.current_plot.add_lines(
		model.shifted_A, model.shifted_b,
		**state.params.plot_options['constraint-contour']
	)
	buffering_plot.add_linf_tr(
		model.shifted_x, model.shifted_r,
		**state.params.plot_options['trust-region'])
	buffering_plot.add_linf_tr(
		model.shifted_x, 1 + state.params.ka,
		**state.params.plot_options['active-region'])
	buffering_plot.add_points(
		br.zs,
		**state.params.plot_options['zik'])
	buffering_plot.add_points(
		br.ws,
		**state.params.plot_options['wik'])
	buffering_plot.add_arrow(
		model.shifted_x, br.u,
		width=0.05 * state.outer_tr_radius,
		**state.params.plot_options['u'])
	buffering_plot.add_point(
		model.shifted_x,
		**state.params.plot_options['current-iterate'])
	for cone in br.cones:
		cone.add_to_plot(buffering_plot, radius=0.25)
	state.buffering_plot = buffering_plot


def compute_buffering_cones(state, model):
	state.logger.start_step('computing buffered region')
	br = BufferedRegion()

	# for when the tr is large...
	buffer_alpha_low = 0.75
	buffer_beta_high = 0.25

	br.bdpb = min(buffer_beta_high, state.params.beta * state.outer_tr_radius ** state.params.p_beta)
	br.adpa = max(buffer_alpha_low, 1 - state.params.alpha * state.outer_tr_radius ** state.params.p_alpha)

	gradients = np.array([ci.evaluate_gradient(model.shifted_x) for ci in model.shifted_constraints])
	gradient_norms = np.linalg.norm(gradients, axis=1)
	constraint_values = np.array([ci.evaluate(model.shifted_x) for ci in model.shifted_constraints])

	zs = []
	for i in range(state.num_constraints):
		if abs(constraint_values[i]) < 1e-12:
			zs.append(model.shifted_x)
		elif abs(gradient_norms[i]) > 1e-12:
			z = -constraint_values[i] / gradient_norms[i] ** 2 * gradients[i]
			zs.append(z)
			assert abs(constraint_values[i] + gradients[i] @ z) < 1e-8, 'could not compute zero of constraint'
			assert abs(model.shifted_A[i] @ z - model.shifted_b[i]) < 1e-8, 'zeros of linearizations did not match'
		else:
			zs.append([np.nan] * state.dim)
	br.zs = np.array(zs)

	# br.ws = model.x + br.adpa * (br.zs - model.x)
	br.ws = br.adpa * br.zs

	constraint_is_active = constraint_values > -1e-8
	constraint_has_gradient = gradient_norms >= 1e-8
	constraint_is_nearly_active = np.max(np.abs(br.zs), axis=1) <= 1 + state.params.ka

	constraint_cone_is_nearly_active = []
	for i in range(state.num_constraints):
		if constraint_is_active[i] or constraint_is_nearly_active[i]:
			constraint_cone_is_nearly_active.append(True)
		elif gradient_norms[i] <= 1e-8:
			constraint_cone_is_nearly_active.append(False)
		else:
			proj, _, _ = project_onto_edge_of_cone(br.ws[i], -gradients[i], br.bdpb, np.zeros_like(model.x))
			constraint_cone_is_nearly_active.append(np.max(np.abs(proj)) <= 1.0)

	br.active_indices = np.logical_or(
		constraint_is_active,
		np.logical_and(
			constraint_has_gradient,
			# constraint_is_nearly_active
			constraint_cone_is_nearly_active))

	br.num_active_constraints = sum(br.active_indices)
	if br.num_active_constraints == 0:
		br.u = np.zeros(state.dim, dtype=np.float64)
		br.u[0] = 1.0
		br.pi = 1.0
	else:
		success, br.u, br.pi = find_feasible_direction(gradients[br.active_indices], state.logger)
	br.pi = min(1.0, br.pi)

	state.logger.info('pi = ' + str(br.pi))

	m = state.num_constraints
	br.cones = [
		Cone.create(br.ws[i], -gradients[i], br.bdpb)
		for i in range(m)
		if br.active_indices[i]
	]

	# this is not defined when the tr is large...
	br.beta0 = br.bdpb + np.sqrt(max(0.0, 1 - br.bdpb ** 2) * max(0.0, 1 - br.pi ** 2))
	br.rot = get_rotation_matrix(br.u)

	state.logger.verbose_json('buffered region', br)
	state.logger.stop_step()

	for cone in br.cones:
		cone.unshift(model.x, model.r).add_to_plot(state.current_plot, radius=0.25 * state.outer_tr_radius)

	br.regular = br.pi > state.params.threshold_regularity

	try:
		create_buffering_plot(state, model, br)
	except:
		print('Unable to plot buffering cones')
		raise

	return br
