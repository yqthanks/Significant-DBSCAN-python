# @Author: xie
# @Date:   2021-06-16T21:19:11-04:00
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2021-06-16T21:19:14-04:00
# @License: MIT License
from sklearn.cluster import DBSCAN
import numpy as np
from numpy.linalg import norm
import math

from helper import *
from montecarlo import *

def get_eps(X, eps_list, minpts_list, i_current, j_current, increase_thrd):
    '''
    Heuristically find which eps to use (for a fixed density), starting from smallest.
    Move to a larger one if the mean size of clusters increases by at least increase_thrd (a proportion of previous mean).
    In general, lower density clusters tend to need a larger eps.
    Here the maximum number of clusters to use for mean-size calculation can be changed by users; defaulted to top 5 clusters.
    '''

    # #for eps, use a relaxed increase_thrd
    # increase_thrd = increase_thrd/2

    max_j = eps_list.shape[0]

    cluster_mean_size_prev = 0
    labels_prev = None

    for j in range(j_current, max_j):
        eps = eps_list[j]
        minpts = minpts_list[i_current,j]

        clusterer = DBSCAN(eps=eps, min_samples=minpts).fit(X)
        unique, counts = np.unique(clusterer.labels_, return_counts=True)
        count_array = np.vstack([unique, counts]).T#get unique labels and counts
        #remove -1 noise points before mean calculation
        cluster_ids = count_array[:,0] >= 0
        count_array = count_array[cluster_ids,:]
        #add sort
        count_array = count_array[np.argsort(count_array[:, 1])]#sort by count (2nd column)
        count_array = count_array[::-1,:]

        if count_array.shape[0] == 0:
            #no clusters
            cluster_mean_size = 0
        else:
            max_to_check = 10
            check_size = min(count_array.shape[0], max_to_check)
            cluster_mean_size = np.mean(count_array[0:check_size,1])
            # cluster_mean_size = np.mean(count_array[:,1])

        if cluster_mean_size_prev == 0:
            cluster_mean_size_prev = cluster_mean_size
            labels_prev = np.array(clusterer.labels_)#might not need the np.array()
        elif (cluster_mean_size - cluster_mean_size_prev) / cluster_mean_size_prev > increase_thrd:
            cluster_mean_size_prev = cluster_mean_size
            labels_prev = np.array(clusterer.labels_)#might not need the np.array()
        else:
            return j-1, labels_prev, cluster_mean_size_prev

        if j == (max_j-1):
            return j, np.array(clusterer.labels_), cluster_mean_size


def get_minpts(X, eps_list, minpts_list, i_current, j_current, increase_thrd, labels_prev, cluster_mean_size_prev):
    '''
    Heuristically find which density (minpts) to use for a fixed eps, starting from the largest.
    (note: part of this function may be merged with get_eps() later)
    '''

    max_i = minpts_list.shape[0]

    # prev results inherited from get_eps()'s outputs
    # cluster_mean_size_prev = 0
    # labels_prev = None
    for i in range(i_current, max_i):
        eps = eps_list[j_current]
        minpts = minpts_list[i,j_current]

        clusterer = DBSCAN(eps=eps, min_samples=minpts).fit(X)
        unique, counts = np.unique(clusterer.labels_, return_counts=True)
        count_array = np.vstack([unique, counts]).T#get unique labels and counts
        #remove -1 noise points before mean calculation
        cluster_ids = count_array[:,0] >= 0
        count_array = count_array[cluster_ids,:]
        #add sort
        count_array = count_array[np.argsort(count_array[:, 1])]#sort by count (2nd column)
        count_array = count_array[::-1,:]

        if count_array.shape[0] == 0:
            #no clusters
            cluster_mean_size = 0
        else:
            max_to_check = 10
            check_size = min(count_array.shape[0], max_to_check)
            cluster_mean_size = np.mean(count_array[0:check_size,1])
            # cluster_mean_size = np.mean(count_array[:,1])

        if cluster_mean_size_prev == 0:
            cluster_mean_size_prev = cluster_mean_size
            labels_prev = np.array(clusterer.labels_)#might not need the np.array()
        elif (cluster_mean_size - cluster_mean_size_prev) / cluster_mean_size_prev > increase_thrd:
            cluster_mean_size_prev = cluster_mean_size
            labels_prev = np.array(clusterer.labels_)#might not need the np.array()
        else:
            return i-1, labels_prev, cluster_mean_size_prev

        if i == (max_i-1):
            return i, np.array(clusterer.labels_), cluster_mean_size


def SigDBSCAN(X, eps_range, density_range, m=100, sig_level=0.01, h=10,
              num_eps = 5, num_density = 5, sample_portion = 0.5,
              increase_thrd = 0.1, complete_random = 2, print_option = 1):
    """
    This version is for arbitrary dimensions, and has not included the grid-based accelerations. Early-termination speed-up is included which mainly works for non-clustered data.
    Consider using a sampled version of the original dataset if it is too large (results may vary).

    For high-dimensional data, it is recommended to use dimension reduction or deep embedding methods to reduce the feature space.

    Paratemers
    -----------
    X: Input data. Feature values should be in range [0,1]

    eps_range: Minimum and maximum eps to consider (X feature value range is in [0,1], so this number needs to be in [0,1])

    density_range: SigDBSCAN enumerates through a list of densities estimated from X.
        This range means the max and min relative density to consider. For example, denote D as the distribution of densities around all points for a given eps,
        a [min, max] = [0.5, 0.9] means CDF(D, min_density) = 0.5, and the max is CDF(D, max_density) = 0.9. CDF is cumulative probability function.

    m: Number of simulation trials, e.g., 19, 99, 999

    sig_level: Significance level, e.g., 0.05, 0.01

    h: number of bins for frequency calculation

    num_eps: number of eps to consider in the eps range

    num_density: number of densities to consider in the density range

    sample_portion: proprtion of data points to use to estimate density distribution (default to 1).
        This is used for both single dimension (for H0 data generation) and all dimensions (used to estimate min and max density for density range)

    increase_thrd: When deciding whether to move to the a larger eps or a lower density
        (choosing (eps, minpts) in a heuristic manner; happens before each significance testing).
        Illustration all candidates for default num_eps = 5 and num_density = 5: (the current search only moves to larger eps or lower density;
        it starts from the next density with the same eps after each significance testing; final result is an aggregation of significant clusters)
                    eps1,               eps2,   ...,    eps5
        density1    (eps1, minpts1),    ...,
        ...         ...,           ,    ...,
        density5    (eps5, minpts5),    ...,

    complete_random: determines the null distribution for each dimension.
        If 1, then H0 data points are completely random in each dimension (e.g., h bins receive equal probability).
        If 0, then probability distribution is estimated using input data X (this might be better for higher dimensions, where data are sparse).
        If 2, then probability distribution is a weighted average of the above two scenarios (weights are defaulted to [0.5, 0.5] and can be changed in estimate_density() function)
        Users can define their own null distribution (using estimate_density() function).
        The null distribution affects the significance of clusters.

    print_option: print key steps and progress if 1; otherwise, no or minimal print
    """
    #data attributes
    n,d = X.shape
    # vis(X)

    y = np.zeros([n]) - 1
    X_id = np.arange(0, n)#data will be updated during Significant DBSCAN, X_id keeps track of indices of data points in the original data
    cluster_id_base = 0 #increase after each round of cluster detetcion (for a density level), making sure no duplicate cluster ids when adding labels of new clusters

    eps_list = np.linspace(eps_range[0], eps_range[1], num=num_eps)
    eps_med = np.median(eps_list)

    #estimate distribution of density (using the median eps to get densities around sample points in data),
    #and generate eps_list and minpts_list as candidates for heuristic search
    density_list_full = np.zeros([n])#use size n first (if n is too large, better first sample points)
    cnt = 0 #counter for number of samples actually used (after rand)
    for i in range(n):
        if np.random.rand() >= sample_portion:
            continue

        center = X[i,:]
        distance = norm(X - center, 2, axis = -1)
        density_list_full[cnt] = np.sum(distance <= eps_med) / (math.pi * (eps_med**2))

        cnt += 1

    density_list_full = np.sort(density_list_full[0:cnt])
    #mode 1: get equal_size bins from min to max density value
    density_min_value = density_list_full[np.floor(density_range[0]*cnt).astype(int)]
    density_max_value = density_list_full[np.floor(density_range[1]*cnt).astype(int)]
    density_list = np.linspace(density_min_value, density_max_value, num=num_density)
    #mode 2: get equal_size bins from min to max density proportion (defined in density_range), and then get the corresponding density values
    # density_range_steps = np.linspace(density_range[0], density_range[1], num=num_density)
    # density_list = np.zeros([num_density])
    # for i in range(num_density):
    #     density_list[i] = density_list_full[np.floor(density_range_steps[i]*cnt).astype(int)]

    #descending order
    density_list = density_list[::-1]

    minpts_list = np.zeros([num_density, num_eps])
    for i in range(num_density):
        for j in range(num_eps):
            minpts_list[i,j] = np.ceil(density_list[i] * math.pi * eps_list[j]**2)

    if print_option == 1:
        print('eps_list: ', eps_list)
        print('minpts_list:\n', minpts_list)

    sig = True
    best_i = 0 #id for minpts_list's dimension 0 (rows)
    best_j = 0 #id for eps_list, starting from smallest (does not decrease for lower density)
    best_y = None
    best_cluster_mean_size_current = 0
    X_id_current = np.copy(X_id)#ids of data points current X (updated in while loop) in the original X
    while sig:
        #data size is updated after removal of points in significant clusters.
        #need to update as this affects H0 data generation in Monte Carlo estimation
        n = X.shape[0]

        #select eps and minpts: heuristic described in paper
        best_j, best_y, best_cluster_mean_size_current = get_eps(X, eps_list, minpts_list, best_i, best_j, increase_thrd)
        if best_i < num_density - 1:
            best_i, best_y, _ = get_minpts(X, eps_list, minpts_list, best_i+1, best_j, increase_thrd, best_y, best_cluster_mean_size_current)
        eps = eps_list[best_j]
        minpts = minpts_list[best_i, best_j]
        if print_option == 1:
            print('\n-----------------------------new round-----------------------------')
            print('Data size (changes after significant cluster removal): ', n)
            print('Selected (eps, minpts) for the current round: (%f, %d)' % (eps, minpts))

        #total number of detected clusters
        num_cluster = max(0, np.max(best_y)+1)
        best_obs_cluster = get_max_cluster_size(best_y)
        if best_obs_cluster == 0:
            print('No clusters detected at the current density level.')
            break

        #significance testing
        #be careful here with the estimation method to use (e.g., completely random or not)
        dist_d = estimate_density(X, h, complete_random)
        thrd = monte_carlo_estimation(n, dist_d, m, sig_level, eps, minpts,
                                      best_obs_cluster = best_obs_cluster, print_option = print_option)
        best_y = remove_spurious_cluster(best_y, thrd)


        if np.max(best_y) < 0:
            if print_option == 1:
                print('Total number of clusters detected: ', num_cluster)
                print('No significant cluster found at the current density level. Continue with next...')
                # print('No significant cluster found at the current density level. Terminating...')
            # use the following to skip the search with lower densities if no significant cluster is found at the current level
            #trade-off: may miss clusters if used, but will take away effect of potential multi-testing (when a large number of densities are used)
            # sig = False
            # break
        else:
            if print_option == 1:
                print('Testing completed for current density level: ')
                print('Total number of clusters detected: ', num_cluster)
                unique, counts = np.unique(best_y, return_counts=True)
                print('Total number of significant clusters: ', np.sum(unique>=0))
                if np.sum(unique>=0) > 0:
                    sig_cluster_sizes = np.sort(counts[unique>=0])
                    print('New significant cluster sizes:', sig_cluster_sizes[::-1])

            #update cluster ids in y
            sig_cluster_list = best_y >= 0
            sig_cluster_original_id = X_id_current[sig_cluster_list]
            y[sig_cluster_original_id] = cluster_id_base + best_y[sig_cluster_list]
            cluster_id_base = np.max(y) + 1

            #update X: remove points in significant clusters
            #note that estimate_distribution() automatically removes space previously taken by the removed points (i.e., reducing freq. in corresponding bins)
            X = X[sig_cluster_list==False]
            X_id_current = X_id_current[sig_cluster_list==False]
            # vis(X)

        if best_i == num_density - 1:
            break
        else:
            best_i += 1

    return y.astype(int)
