import numpy as np
from scipy import sparse

def mask_nan_entries_with_avg(adj_matrix):
    # the nan entries are masked
    mask = np.isnan(adj_matrix)
    masked_arr = np.ma.masked_array(adj_matrix, mask)
    movie_means = np.mean(masked_arr, axis=0)

    # nan entries will replaced by the average rating for each item
    adj_matrix = masked_arr.filled(movie_means)
    return adj_matrix

def find_max_edge(derivative_network):
    """Finds the maximumum user-item relationship (rating) of the derivative network
    (A bit slow might need modifications)
    """
    max_ind = (-1, -1)
    max_val = float('-inf')
    derivative_network = sparse.coo_matrix(derivative_network)
    for i, j, derivative in zip(derivative_network.row, derivative_network.col, derivative_network.data):
        if derivative > max_val:
            max_ind = (i, j)
            max_val = derivative
    return max_ind