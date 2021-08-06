import sys
import random
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split

from utils import *
from load_data import *

""" Example Run : python(3) n2n.py {jester, movielens} """

def svd(adj_matrix, k, mask):
    """Computes the svd of adj_matrix

    Args:
        adj_matrix:
        k: the number of latent features to keep

    Returns:
        UsV: computed SVD form of the adjacency matrix
    """
    if mask == True:
        adj_matrix = mask_nan_entries_with_avg(adj_matrix)

    adj_matrix = sparse.coo_matrix(adj_matrix)
    U, s, V = sparse.linalg.svds(adj_matrix, k=k)
    s = np.diag(s)
    UsV = (U.dot(s)).dot(V)
    return UsV

def calculate_derivative_network(adj_matrix, k, mask):
    """Computes the derivative network of the graph

    Args:
        adj_matrix: adjacency matrix of the graph
        k: number of latent factors to keep

    Returns:
        derivative_network: the n2n derivative network of the graph
    """
    if mask == True:
        adj_matrix = mask_nan_entries_with_avg(adj_matrix)

    U, s, Vh = np.linalg.svd(adj_matrix, full_matrices=False)
    s = np.diag(s)
    derivative_network = 2 * U[:, k+1:] @ s[k+1:, k+1:] @ Vh[k+1:, :]
    return derivative_network

def predict_ratings():
    return None

def perturb_matrix(adj_matrix, num_perturbs, k, mask):
    """Removes the edges with maximum value in the derivative network

    Args:
        adj_matrix: adjacency matrix of the graph
        num_perturbs: number of perturbations to perform
        k: number of latent factors to keep

    Returns:
        perturbed_adj_matrix: the perturbed adjacency matrix
    """
    perturbed_adj_matrix = np.copy(adj_matrix)
    for i in range(num_perturbs):
        print("Perturb Iteration:", i)
        derivative_network = calculate_derivative_network(perturbed_adj_matrix, k, mask)
        max_edge = find_max_edge(derivative_network)
        if mask == True:
            perturbed_adj_matrix[max_edge[0]][max_edge[1]] = np.nan
        else:
            perturbed_adj_matrix[max_edge[0]][max_edge[1]] = 0

    return perturbed_adj_matrix

def evaluate_error(orig_svd, perturbed_svd, orig_adj_matrix, omega_c):
    """RMSE

    Args:
        orig_svd: predictions made by the original model
        perturbed_svd: prediction made after perturbation
        orig_adj_matrix: adjacency matrix of the graph with NaN values
        omega_c: the cardinality of the complement of the omega set
    
    Returns:
        error: the rmse error
    """
    error = 0
    num_rows, num_cols = orig_svd.shape[0], orig_svd.shape[1]
    for i in range(num_rows):
        for j in range(num_cols):
            if np.isnan(orig_adj_matrix[i, j]) or orig_adj_matrix[i, j] == 0:
                error += (orig_svd[i, j] - perturbed_svd[i, j])**2
    error = error / omega_c
    error = np.sqrt(error)
    return error

def main(argv):
    if len(argv) < 1:
        print("Please mention a dataset (movielens or jester)")
        return

    k = 5
    num_perturbs = 20
    dataset_name = argv[0]

    mask = False
    if dataset_name == "movielens" or dataset_name == "jester" or dataset_name == "modcloth":
        mask = True
    
    adj_matrix = load_data(dataset_name)
    perturbed_matrix = perturb_matrix(adj_matrix, num_perturbs, k, mask)

    omega_c = 0
    if mask == True:
        omega_c = np.count_nonzero(np.isnan(adj_matrix))
    else:
        omega_c = np.count_nonzero(adj_matrix == 0)

    orig_svd = svd(adj_matrix, k, mask)
    perturbed_svd = svd(perturbed_matrix, k, mask)

    error = evaluate_error(orig_svd, perturbed_svd, adj_matrix, omega_c)
    print("RMSE Error:", error)
    
if __name__ == "__main__":
    main(sys.argv[1:])