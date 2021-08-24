# Always Feasible DFO

This is an Algorithm for solving constrained derivative-free optimization problems.
Specifically, this algorithm is designed to maintain feasibility of iterates and sample points.

<hr>

## Installing Dependencies


### Anaconda
This code uses the anaconda software library https://www.anaconda.com/.
Installing Anaconda will allow you to install the required dependencies:

```
conda install -c conda-forge pyomo matplotlib scipy ipopt cython
```

### PDFO

Also, we use pdfo which is available at https://www.pdfo.net/:

```
python -m pip install pdfo
```

This library implements derivative-free algorithms created M. J. D. Powell.


### NOMAD

NOMAD is a direct-search, derivative-free algorithm available at https://www.gerad.ca/nomad/.
This library must be compiled to be called from within our algorithm.


### Hock-Schittkowski

We implemented a wrapper around the Hock-Schittkowski test problems for non-linear optimization.
The problem set is described in https://www.springer.com/gp/book/9783540105619.
The source code can be downloaded from http://klaus-schittkowski.de/downloads.htm.

After the source is downloaded, it can be built within the schittkoski_library Makefile.


### Configuration


Once installed, please modify settings.py to direct the algorithm's output to the desired directory.
- Set OUTPUT_DIRECTORY to the location you would like to view results and iteration plots in.
- Set IP_OPT to the location of the ipopt executable installed through Anaconda.
- Set SCHITTKOWSKI_LIBRARY to the location you compiled the Fortran library.



## Organization

The code is roughly organized as follows:

algorithm/
This contains most of the core algorithm components.
there are two subfolders for different variants of the sample region and trust region subproblems used.


driver/
This contains the files to call the algorithm on with different parameters
- run_hock_schittkowski.py runs our algorithm on the hock-schittkowski problem set.
- compare_schittkowski.py compares our library implementation to the 2011 Fortran code.
- create_history.py creates a plot of the evaluations for a particular run
- create_performance_plot.py creates a performance plot for all output directories in an output folder (the folder must be edited)
- run_nomad.py, run_pdfo.py, and run_schipy.py run other libraries on the hock-schittkowski problem set


hock_schittkowski/
This contains an implementation of several of the problems within the Hott-Schittkowski problem set.

pyomo_opt/
This contains calls to the Pyomo library that are used as subroutines in our algorithm

utils/
This contains various helpers used within our algorithm.

visualizations/
This contains code to create plots of various subroutines to better understand how they work.


