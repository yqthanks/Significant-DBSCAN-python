# @Author: xie
# @Date:   2021-06-16T21:28:19-04:00
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2021-06-16T21:28:21-04:00
# @License: MIT License

import numpy as np

def get_max_cluster_size(cluster_labels):
    unique, counts = np.unique(cluster_labels, return_counts=True)
    count_array = np.vstack([unique, counts]).T#get unique labels and counts
    count_array = count_array[np.argsort(count_array[:, 1])]#sort by count (2nd column)
    count_array = count_array[::-1,:]#reverse the sort order --> descending
    #get the max size
    max_size = 0
    if count_array[0,0] == -1:#label "-1" means noise
        if count_array.shape[0] > 1:
            max_size = count_array[1,1]
    else:
        max_size = count_array[0,1]
    return max_size

def remove_spurious_cluster(cluster_labels, threshold):
    unique, counts = np.unique(cluster_labels, return_counts=True)
    count_array = np.vstack([unique, counts]).T#get unique labels and counts

    spurious_cluster_ids = count_array[np.where(count_array[:,1]<=threshold), 0]#cluster ids
    spurious_points = np.isin(cluster_labels, spurious_cluster_ids)#true or false (this list has equal size to cluster_labels, so true/false can be used to select rows)
    cluster_labels[spurious_points] = -1

    return cluster_labels
