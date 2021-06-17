# @Author: xie
# @Date:   2021-06-16T20:49:35-04:00
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2021-06-16T20:49:45-04:00
# @License: MIT License

import numpy as np
from sigdbscan import *
from data import *

#load data
#three example demo data (more in sample_data):
#1. shapedata_example_1.txt (N=10,000; clustered data; clusters with very different densities)
#2. shapedata_example_2.txt (N=4,000; clustered data; clusters with similar densities)
#3. randomdata_example.txt (N=6,000; non-clustered data; purely random)
file_path = 'sample_data/demo/shapedata_example_2.txt'
X = np.loadtxt(file_path, delimiter=',')

#normalize and visualize data
#by default, values along all dimensions will be normalized to [0,1] ranges
#the normalize_data() function allows different min,max bounds across dimensions
#(still need to make sure normalized feature values are in [0,1] range)
X = normalize_data(X)
# vis(X)#only for 1D/2D

#define ranges for eps and density_range
#see definitions in SigDNSCAN.py (currently eps is isotropic for different dimensions; will update later)
eps_range = np.array([0.02, 0.05])#min & max eps
density_range = np.array([0.25, 0.75])#min & max density (percentile in overall density distribution in data)

#run SigDBSCAN (a simplified version without optimization and accelerations for low dimensional data)
#see SigDNSCAN.py for parameter definitions
#not recomending increasing number_density by too much, otherwise may introduce multi-testing issue;
#if large num_density is used, to avoid/mitigate the potential multi-testing issue, uncomment the "break" in SigDBSCAN() to terminate the search if no significant cluster is found at a density level
#change print_option to 0 to hide intermediate prints
y = SigDBSCAN(X, eps_range, density_range, m=100, sig_level=0.01, h=10,
              num_eps = 5, num_density = 5, sample_portion = 1,
              increase_thrd = 0.1, complete_random = 2, print_option = 1)

#visualize results (noise points by default have (1,1,1) color)
vis(X, y, vis_noise = True)
