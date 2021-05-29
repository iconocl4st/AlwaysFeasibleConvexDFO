
import numpy as np
import os

from settings import EnvironmentSettings
from utils.plotting import Plotting
from utils.convex_hull import get_convex_hull
from utils.polyhedron import Polyhedron


def visualize_convex_hull(points):
	A, b, indices = get_convex_hull(points)
	poly = Polyhedron(A, b)

	filename = EnvironmentSettings.get_sub_path('visualizations/convex_hull/convex_hull.png')
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	plot = Plotting.create_plot_on(filename, [-1.2, -1.2], [1.2, 1.2], name='convex hull')
	plot.add_points(points, label='points')
	plot.add_polyhedron(poly, label='convex hull')
	plot.save()


if __name__ == '__main__':
	visualize_convex_hull(2 * np.random.random([10, 2]) - 1)
