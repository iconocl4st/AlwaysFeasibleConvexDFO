

This code uses the anaconda software library https://www.anaconda.com/.
Installing anaconda will allow you to install the required dependencies:

```
conda install -c conda-forge pyomo matplotlib scipy ipopt cython
```

Also, we use pdf available at https://www.pdfo.net/:

python -m pip install pdfo

Once installed, please modify settings.py to direct the algorithm's output to the desired directory.
Set OUTPUT_DIRECTORY to the location with you would like to view results and iteration plots in.
Set IP_OPT to the location of the ipopt executable installed through Anaconda.
Set SCHITTKOWSKI_LIBRARY to the locaiton you compiled the Fortran library.







The code is roughly organized as follows:

algorithm/
This contains most of the core algorithm components.
there are two subfolders for different variants of the sample region and trust region subproblems used.


driver/
This contains the files to call the algorithm on with different parameters
run_hock_schittkowski.py runs our algorithm on the hott-schittkowski problem set.
compare_schittkowski.py compares our library implementation to the 2011 fortran code.
create_history.py creates a plot of the evaluations for a particular run
create_performance_plot.py create a performance plot for all output directories in an output folder
run_nomad.py, run_pdfo.py, and run_schipy.py run other libraries on the hott-schittkowski problem set


hock_schittkowski/
This contains an implementation of several of the problems within the Hott-Schittkowski problem set.

pyomo_opt/
This contains calls to the Pyomo library that are used as subroutines in our algorithm

utils/
This contains various helpers used within our algorithm.

visualizations/
This contains code to create plots of various subroutines to better understand how they work.
