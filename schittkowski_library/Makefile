

# EDIT PROB.FOR
#	add a comma to the end of line 7219
#	add a space after the / on line 5589

probs:
	gfortran -c PROB_MOD.FOR

conv:
	gfortran -c CONV.FOR

library:
	python -m numpy.f2py -c PROB.FOR CONV.FOR  -m schittkowski_library 

all: probs conv
	
	
	

import schittkowski_library as schit
