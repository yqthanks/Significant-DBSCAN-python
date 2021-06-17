# @Author: xie
# @Date:   2021-06-16T21:32:45-04:00
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2021-06-16T21:33:02-04:00
# @License: MIT License

from sklearn.cluster import DBSCAN
import numpy as np
from numpy.linalg import norm
import math

from helper import *

def estimate_density(X, h, complete_random = 2):
    """Estimate frequency distribution along each dimension.
    Assuming dimensions are independent. Preprocessing (e.g., dimension reduction)
    are recommended for data containing correlated dimensions.
    Note: Users can define their own null frequency distribution by customizing this function.

    Paratemers
    -----------
    X: Input data. Feature values should be in range [0,1]

    h: number of bins for frequency calculation

    complete_random: determines the null distribution for each dimension.
        If 1, then H0 data points are completely random in each dimension (e.g., h bins receive equal probability).
        If 0, then probability distribution is estimated using input data X (this might be better for higher dimensions, where data are sparse).
        If 2, then probability distribution is a weighted average of the above two scenarios (weights are defaulted to [0.5, 0.5] and can be changed in estimate_density() function)
        Users can define their own null distribution (using estimate_density() function).
        The null distribution affects the significance of clusters.

    Returns
    -------
    dist_d: Numpy array. Distribution (freq.) of data points along each dimension
    (distribution along each of the d dimensions is represented as a vector with h bins).
    Size: d x h, can be user-defined
    """

    n, d = X.shape#number of points and dimensions
    dist_d = np.zeros([d,h])

    if complete_random == 1:
        dist_d = dist_d + 1/h
        return dist_d

    bin_width = 1/h
    X_bin = np.floor(X / bin_width).astype(int)
    #make max value to h-1 bin (or use a more efficient fix by adding a very small value to max for each dimension)
    X_bin[X_bin == h] = h - 1

    for i in range(d):
        unique, counts = np.unique(X_bin[:,i], return_counts=True)
        for j in range(unique.shape[0]):
            bin_id = unique[j]
            dist_d[i, bin_id] = counts[j] / n

    if complete_random == 2:
        dist_d_random = 1/h
        data_weight = 0.75
        random_weight = 1 - data_weight
        dist_d = data_weight * dist_d + random_weight * dist_d_random

    return dist_d

def generate_H0_data(n, dist_d):
    """Estimate test statistic for spurious cluster removal.

    Paratemers
    -----------
    n: Data size
    dist_d: Numpy array. Distribution (freq.) of data points along each dimension
    (distribution along each dimension is represented as a vector with h bins). Example: [[0.4, 0.6], [0.1, 0.9]]
    In addition, the range of valid values (not freq.) along each dimension is assumed to be in [0,1].
    """
    d = dist_d.shape[0]#number of dimensions
    h = dist_d.shape[1]#number of bins to calculate the frequency distribution along each dimenison

    h0_data = np.zeros([n,d])
    #get the min value for each bin;again, range of valid values assumed to be in [0,1]
    bin_base = np.arange(h)/h
    for i in range(d):
        #get bin selections randomly for each point
        bin_random = np.random.choice(h, n, p=dist_d[i,:])
        # h0_data[:,i] = np.reshape(bin_base[bin_random], [-1,1]) + np.random.rand(n,1)/h
        h0_data[:,i] = bin_base[bin_random] + np.random.rand(n)/h

    return h0_data

def monte_carlo_estimation(n, dist_d, m, sig_level, eps, minpts, h=10, best_obs_cluster = np.inf, print_freq = 10, print_option = 1):
    """Estimate test statistic for spurious cluster removal.

    Paratemers
    -----------
    X: Input data. Feature values should be in range [0,1]

    h: number of bins for frequency calculation

    dist_d: Numpy array. Distribution (freq.) of data points along each dimension (distribution along each dimension is represented as a vector with h bins). Example: [[0.4, 0.6], [0.1, 0.9]]

    m: Number of simulation trials, e.g., 19, 99, 999

    sig_level: Significance level, e.g., 0.05, 0.01

    eps: eps for DBSCAN (must be the same for observed data)

    minpts: minpts for DBSCAN (must be the same for observed data)

    best_obs_cluster: size of the largest cluster detected from observed data; if a value is provided, will be used to enable early-termination

    print_freq: how often does the function print about the trial id being executed (help user to estimate time left)
    """

    if print_option == 1:
        print('Monte Carlo estimation started (may take some time to finish)...')
        print('Total trial number: ', m)

    monte_carlo_table = np.zeros(m)
    early_term_cnt = 0
    for i in range(m):
        h0_data = generate_H0_data(n, dist_d)
        clusterer = DBSCAN(eps, min_samples=minpts).fit(h0_data)
        monte_carlo_table[i] = get_max_cluster_size(clusterer.labels_)
        if monte_carlo_table[i] >= best_obs_cluster:
            early_term_cnt += 1
            if early_term_cnt >= np.ceil(m * sig_level):
                if print_option == 1:
                    print("Terminated early: no significant clusters...")
                return np.inf

        if i % print_freq == 0:
            if print_option == 1:
                # print('Trial ', i, ', ', m-i-1, ' trials to complete...')
                print(i, 'trials completed...')

    monte_carlo_table = np.sort(monte_carlo_table)#sort
    monte_carlo_table = monte_carlo_table[::-1]#reverse order --> descending
    idx = np.ceil(m * sig_level).astype(int)
    threshold = monte_carlo_table[idx]

    # print(monte_carlo_table)

    return threshold
