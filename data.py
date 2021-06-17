# @Author: xie
# @Date:   2021-06-16T21:25:21-04:00
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2021-06-16T21:25:54-04:00
# @License: MIT License

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_data(X, range_d = None):
    """
    Ideally data should be normalized by users during preprocessing.
    This is only used to make sure all feature values are in a [0,1] range.

    Parameters:
    -----------
    X: input data with d features
    range_d: A 2xd array storing the min and max range value used to normalize each dimension.
        Note that the min, max do not need to be the min, max values obtained from data, especially
        if value ranges should be different across dimensions.
        If not provided, min, max values for each dimension in the data will be used for normalization.
    """
    n,d = X.shape

    if range_d is None:
        range_d = np.zeros([2,d])
        range_d[0,:] = np.min(X, axis = 0)
        range_d[1,:] = np.max(X, axis = 0)

    X = (X - range_d[0,:]) / (range_d[1,:] - range_d[0,:])

    return X

def vis(X, y = None, vis_noise = False):
    """
    Two (or one) dimensions only.
    Subspaces are needed for visualizing higher dimensional data.
    """
    plt.figure()

    if y is None:
        plt.scatter(*X.T, s=1)
    else:
        color_noise = (1,1,1)
        if vis_noise:
            color_noise = (0.75, 0.75, 0.75)

        color_palette = sns.color_palette('deep', np.max(y).astype(int)+1)
        cluster_colors = [color_palette[y_i] if y_i >= 0
                        else color_noise
                        for y_i in y]

        plt.scatter(*X.T, s=1, c=cluster_colors)

    plt.show()
