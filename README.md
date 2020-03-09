Lefschetz thimble method applied to complex scalar field of dimension 1 under chemical potential on a lattice

The code is based the general method presented in arXiv:1510.03258.

The main part is the file phi4mu.py. It has to be launched as

python3 phi4mu.py mu T

where mu is the value of the chemical potential in unit of mass. (between 0 and 1) and T is the flow time integration.

The file data_analysis.py deals with the data analysis through Jackknife method. At the same time, it produces associated graph representing the iamginary part of the action with respect to the flow time and the number of particle expectation value with respect to the chemical potential.
