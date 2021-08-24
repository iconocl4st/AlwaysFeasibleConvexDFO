import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

import numpy as np
import os

from utils.bounds import Bounds
from settings import EnvironmentSettings

''' TODO: add _ '''


def get_other_unit_vectors(direction, beta):
	# TODO: This could be simplified...
	d = direction / np.linalg.norm(direction)
	vecx1 = beta * d[0] + d[1] * np.sqrt(1 - beta * beta)
	vecx2 = beta * d[0] - d[1] * np.sqrt(1 - beta * beta)

	vecy1 = beta * d[1] + d[0] * np.sqrt(1 - beta * beta)
	vecy2 = beta * d[1] - d[0] * np.sqrt(1 - beta * beta)

	vec1 = np.array([vecx1, vecy1])
	vec2 = np.array([vecx1, vecy2])
	vec3 = np.array([vecx2, vecy1])
	vec4 = np.array([vecx2, vecy2])

	# is it always vec2 and vec3?

	actual_answers = []
	if np.abs(beta - d @ vec1) < 1e-4:
		actual_answers.append(vec1)
	if np.abs(beta - d @ vec2) < 1e-4:
		actual_answers.append(vec2)
	if np.abs(beta - d @ vec3) < 1e-4:
		actual_answers.append(vec3)
	if np.abs(beta - d @ vec4) < 1e-4:
		actual_answers.append(vec4)
	return actual_answers


def get_angle(direction):
	d = direction / np.linalg.norm(direction)
	angle = np.arccos(d @ np.array([1, 0]))
	degrees = angle * 180 / np.pi
	if d @ np.array([0, 1]) >= 0:
		return degrees
	else:
		return 360 - degrees


class MultiPlot:
	def __init__(self):
		self.plots = []

	def add_plot(self, p):
		self.plots.append(p)

	def save(self):
		for p in self.plots:
			p.save()

	def add_linf_tr(self, center, radius, label, color='b', width=1):
		for p in self.plots:
			p.add_linf_tr(center, radius, label, color, width)

	def add_line(self, a, b, label, color='b'):
		for p in self.plots:
			p.add_line(a, b, label, color)

	def add_lines(self, a, b, label, color='b'):
		for p in self.plots:
			p.add_lines(a, b, label, color)

	def add_contour(self, func, label, color='k', lvls=None):
		for p in self.plots:
			p.add_contour(func, label, color, lvls)

	def add_points(self, points, label, color='r', s=20, marker="x"):
		for p in self.plots:
			p.add_points(points, label, color, s, marker)

	def add_point(self, point, label, color='r', s=20, marker="x"):
		for p in self.plots:
			p.add_point(point, label, color, s, marker)

	def add_arrow(self, x1, x2, color="red", width=0.05, label='label'):
		for p in self.plots:
			p.add_arrow(x1, x2, color, width, label)

	def add_polyhedron(self, polyhedron, label, color='b', lvls=[-0.1, 0.0]):
		for p in self.plots:
			p.add_polyhedron(polyhedron, label, color, lvls)

	def add_wedge(self, vertex, direction, beta, radius, label, color=None):
		for p in self.plots:
			p.add_wedge(vertex=vertex, direction=direction, beta=beta, radius=radius, label=label, color=color)


class NoPlot:
	def add_circle(self, *args, **kwargs):
		pass

	def add_plot(self, *args, **kwargs):
		pass

	def save(self):
		pass

	def add_linf_tr(self, *args, **kwargs):
		pass

	def add_contour(self, *args, **kwargs):
		pass

	def add_points(self, *args, **kwargs):
		pass

	def add_point(self, *args, **kwargs):
		pass

	def add_arrow(self, *args, **kwargs):
		pass

	def add_polyhedron(self, *args, **kwargs):
		pass

	def add_wedge(self, *args, **kwargs):
		pass

	def add_line(self, *args, **kwargs):
		pass

	def add_lines(self, *args, **kwargs):
		pass


'''
TODO: fix unused labels/arguments...
'''


class PlotObject:
	def __init__(self):
		self.bounds = None
		self.ax = None
		self.fig = None
		self.filename = None
		self.x = None
		self.y = None
		self.X = None
		self.Y = None
		self.Z = None
		self.css = []

	def save(self):
		for cs in self.css:
			plt.clabel(cs, fontsize=9, inline=1)
		# artists, labels = cs.legend_elements()
		# self.ax.legend()
		self.ax.legend()
		self.ax.grid(True)
		# print('saving to {}'.format(os.path.abspath(self.filename)))
		os.makedirs(os.path.dirname(self.filename), exist_ok=True)
		self.fig.savefig(self.filename)
		plt.close()

	def add_vectorized_contour(self, func, label, color='k', lvls=None):
		assert False, 'implement me'

	def add_lines(self, a, b, label, color='b'):
		if label is not None:
			first_label = label + ' [0-' + str(a.shape[0]) + ']'
		else:
			first_label = None
		for i, (ai, bi) in enumerate(zip(a, b)):
			self.add_line(
				ai, bi,
				label=first_label if i == 0 else None,
				color=color)

	def add_line(self, a, b, label, color='b'):
		nrm = np.linalg.norm(a)
		if nrm < 1e-12:
			print('normal for line is zero, skipping')
		a = a.copy() / nrm
		b = b / nrm
		if abs(a[1]) < 1e-8:
			# vertical line, a[0] * x[0] == b
			plt.vlines(b / a[0], ymin=self.bounds.lb[1], ymax=self.bounds.ub[1], label=label, colors=[color])
		else:
			# a[0] * x[0] + a[1] * x[1] == b
			# x[1] == (b - a[0] * x[0]) / a[1]
			xmin = self.bounds.lb[0]
			xmax = self.bounds.ub[0]
			plt.plot(
				[xmin, xmax],
				[(b - a[0] * xmin) / a[1], (b - a[0] * xmax) / a[1]],
				label=label,
				color=color
			)

	def add_linf_tr(self, center, radius, label, color='b', width=1):
		self.ax.add_patch(patches.Rectangle(
			center - radius,
			2 * radius,
			2 * radius,
			label=label,
			linewidth=width,
			edgecolor=color,
			facecolor='none'))

	def add_contour(self, func, label, color='k', lvls=None):
		for i in range(0, len(self.x)):
			for j in range(0, len(self.y)):
				self.Z[j, i] = func(np.array([self.x[i], self.y[j]]))
		if lvls is None:
			cs = plt.contour(self.X, self.Y, self.Z, colors=color)
			self.css.append(cs)
		else:
			cs = plt.contour(self.X, self.Y, self.Z, levels=lvls, colors=color)
			self.css.append(cs)
		if len(cs.collections) > 0 and label is not None:
			cs.collections[-1].set_label(label)

	def add_points(self, points, label, color='r', s=20, marker="x"):
		if len(points) == 0:
			return
		self.ax.scatter(points[:, 0], points[:, 1], s=s, c=color, marker=marker, label=label)

	def add_point(self, point, label, color='r', s=20, marker="x"):
		self.ax.scatter([point[0]], [point[1]], s=s, c=color, marker=marker, label=label)

	def add_circle(self, center, radius, color='b', label=None):
		self.ax.add_artist(plt.Circle(center, radius, color=color, fill=False))

	def add_arrow(self, x1, x2, color="red", width=0.05, label='label'):
		if width is None:
			self.ax.add_patch(patches.Arrow(
				x=x1[0], y=x1[1],
				dx=x2[0] - x1[0], dy=x2[1] - x1[1],
				facecolor=color,
				edgecolor=color,
				label=label
			))
		else:
			self.ax.add_patch(patches.Arrow(
				x=x1[0], y=x1[1],
				dx=x2[0] - x1[0], dy=x2[1] - x1[1],
				facecolor=color,
				edgecolor=color,
				width=width,
				label=label
			))

	def add_polyhedron(self, polyhedron, label, color='b'):
		self.add_lines(polyhedron.A, polyhedron.b, color=color, label=label)

	def add_wedge(self, vertex, direction, beta, radius, color, label=None):
		others = get_other_unit_vectors(direction, beta)
		a1 = get_angle(others[0])
		a2 = get_angle(others[1])
		min_a = min(a1, a2)
		max_a = max(a1, a2)
		if max_a - min_a <= 180:
			pa1 = min_a
			pa2 = max_a
		else:
			pa1 = max_a - 360
			pa2 = min_a
		self.ax.add_patch(patches.Wedge(
			vertex, radius,
			pa1, pa2,
			color=color
			# , edgecolor='b', facecolor='none'
		))


class ThreeDPlot(PlotObject):
	def __init__(self):
		self.bounds = None
		self.ax = None
		self.fig = None
		self.filename = None
		self.x = None
		self.y = None
		self.X = None
		self.Y = None
		self.Z = None
		self.css = []

	def save(self):
		for cs in self.css:
			plt.clabel(cs, fontsize=9, inline=1)
		self.ax.legend()
		self.ax.grid(True)
		os.makedirs(os.path.dirname(self.filename), exist_ok=True)
		self.fig.savefig(self.filename)
		plt.close()

	def add_point(self, point, label, color='r', s=20, marker="x"):
		self.add_points(point[np.newaxis, :], label=label, color=color, s=s, marker=marker)

	def add_points(self, points, label, color='r', s=20, marker="x"):
		self.ax.scatter3D(
			points[:, 0],
			points[:, 1],
			points[:, 2],
			s=s, c=color,
			marker=marker,
			label=label)

	def add_line(self, a, b, label, color='b'):
		if abs(a[2]) < 1e-8:
			return

		xx, yy = np.meshgrid(
			np.linspace(self.bounds.lb[0], self.bounds.ub[0], 10),
			np.linspace(self.bounds.lb[1], self.bounds.ub[1], 10))
		zz = (b - (a[0] * xx + a[1] * yy)) / a[2]
		c1 = self.ax.plot_surface(xx, yy, zz, alpha=0.2, label=label, color=color)
		c1._facecolors2d = c1._facecolors
		c1._edgecolors2d = c1._edgecolors

	def add_linf_tr(self, center, radius, label, color='b', width=1):
		print('implement me')

	def add_contour(self, func, label, color='k', lvls=None):
		print('implement me')

	def add_circle(self, center, radius, color='b', label=None):
		print('implement me')

	def add_arrow(self, x1, x2, color="red", width=0.05, label='label'):
		print('implement me')

	def add_wedge(self, vertex, direction, beta, radius, color, label=None):
		print('implement met')


class Plotting:
	@staticmethod
	def create_plot(title, filename, bounds):
		n = len(bounds.lb)

		if n == 2:
			ret_val = PlotObject()
		elif n == 3:
			# ret_val = ThreeDPlot()
			return NoPlot()
		else:
			return NoPlot()

		ret_val.fig = plt.figure()
		if n == 2:
			ret_val.ax = ret_val.fig.add_subplot(111)
		elif n == 3:
			ret_val.ax = plt.axes(projection='3d')

		ret_val.bounds = bounds
		ret_val.filename = filename
		plt.legend(loc='lower left')
		PLOT_SIZE = 10
		scale_factor = max(0.25, min(4.0, (bounds.ub[1] - bounds.lb[1]) / (bounds.ub[0] - bounds.lb[0])))
		ret_val.fig.set_size_inches(PLOT_SIZE, scale_factor * PLOT_SIZE)
		plt.title(title)

		if n == 2:
			matplotlib.rcParams['xtick.direction'] = 'out'
			matplotlib.rcParams['ytick.direction'] = 'out'

			plt.ylim(bounds.lb[1], bounds.ub[1])
			plt.xlim(bounds.lb[0], bounds.ub[0])

			ret_val.x = np.linspace(bounds.lb[0], bounds.ub[0], num=100)
			ret_val.y = np.linspace(bounds.lb[1], bounds.ub[1], num=100)
			ret_val.X, ret_val.Y = np.meshgrid(ret_val.x, ret_val.y)
			ret_val.Z = np.zeros((len(ret_val.y), len(ret_val.x)))
		return ret_val

	@staticmethod
	def create_plot_on(filename, lb, ub, name='a plot'):
		return Plotting.create_plot(name, filename, Bounds.create(lb, ub))

	@staticmethod
	def combine_plots(plts):
		mp = MultiPlot()
		for plt in plts:
			mp.add_plot(plt)
		return mp


class Plotter:
	def __init__(self):
		self.index = 0
		self.path = None

	def get_next_plot_path(self, filename, subfolder=None):
		p = self.path
		if subfolder is not None:
			p = os.path.join(p, subfolder)
		p = os.path.join(p, str(self.index).zfill(5) + '_' + filename + '.png')
		os.makedirs(os.path.dirname(p), exist_ok=True)
		self.index += 1
		return p

	def create_plot(self, filename, bounds, title, subfolder=None):
		return Plotting.create_plot(title, self.get_next_plot_path(filename, subfolder), bounds)

	def to_json(self):
		return {'current-plot-count': self.index, 'image-path': self.path}

	@staticmethod
	def parse_json(json, new_image_path=None):
		plotter = Plotter()
		plotter.index = json['current-plot-count']
		plotter.path = json['image-path']
		if new_image_path is not None:
			plotter.path = new_image_path
		return plotter

	@staticmethod
	def create(root_directory):
		plotter = Plotter()
		plotter.path = root_directory
		return plotter


if __name__ == "__main__":
	p = Plotting.create_plot_on(
		'./testing.png',
		[-2, -2, -2],
		[2, 2, 2],
		name='a plot')

	# Data for a three-dimensional line
	# zline = np.linspace(0, 15, 1000)
	# xline = np.sin(zline)
	# yline = np.cos(zline)xd
	# ax.plot3D(xline, yline, zline, 'gray')
	#
	# Data for three-dimensional scattered points
	zdata = 15 * np.random.random(100)
	xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
	ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
	points = np.array([xdata, ydata, zdata]).T
	p.add_points(points, label='some points')
	p.save()
