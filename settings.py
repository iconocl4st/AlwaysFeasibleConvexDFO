import os
import sys
import traceback
import shutil

import numpy as np

'''
To set up the environment:
conda install -c conda-forge pyomo matplotlib scipy ipopt cython

edit src/Double.cpp:133
NOMAD_HOME=/work/research/nomad/nomad.3.9.1 ./configure
NOMAD_HOME=/work/research/nomad/nomad.3.9.1 make -j9
pushd examples/interfaces/pyNomad_Beta
NOMAD_HOME=/work/research/nomad/nomad.3.9.1 python setup_PyNomad.py install
popd


python -m pip install pdfo

'''

np.set_printoptions(linewidth=255)


class TracePrints(object):
	def __init__(self):
		self.stdout = sys.stdout

	def write(self, s):
		self.stdout.write("Writing %r\n" % s)
		traceback.print_stack(file=self.stdout)


class EnvironmentSettings:
	OUTPUT_DIRECTORY = '/home/thallock/Pictures/ConvexConstraintsOutput'

	SOLVER = 'ipopt'
	IP_OPT = '/mnt/1f0ab4b3-c472-49e1-92d8-c0b5664f7fdb/anaconda3/envs/ConvexConstraints/bin/ipopt'
	# SOLVER_PATH = '/home/thallock/Applications/cplex/cplex/bin/x86-64_linux/cplex'


	SCHITTOWSKI_LIBRARY = '/work/research/schittowski_library/install'

	@staticmethod
	def remove_files(root):
		for root, _, files in os.walk(root):
			for file in files:
				child_path = os.path.join(root, file)
				print('Deleting ' + child_path)
				os.remove(child_path)

	@staticmethod
	def remove_images(root):
		for root, _, files in os.walk(root):
			for file in files:
				if not file.endswith('.png'):
					continue
				child_path = os.path.join(root, file)
				print('Deleting ' + child_path)
				os.remove(child_path)
		# shutil.rmtree(subpath, ignore_errors=True)

	@staticmethod
	def get_output_path(paths):
		return os.path.join(*([EnvironmentSettings.OUTPUT_DIRECTORY] + paths))

	@staticmethod
	def find_print_statements():
		sys.stdout = TracePrints()
