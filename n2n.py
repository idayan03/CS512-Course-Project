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

def svd(adj_matrix, k):
    """Computes the svd of adj_matrix

    Args:
        adj_matrix:
        k: the number of latent features to keep

    Returns:
        UsV: computed SVD form of the adjacency matrix
    """
    adj_matrix = mask_nan_entries_with_avg(adj_matrix)
    adj_matrix = sparse.coo_matrix(adj_matrix)
    U, s, V = sparse.linalg.svds(adj_matrix, k=k)
    s = np.diag(s)
    UsV = (U.dot(s)).dot(V)
    return UsV

def calculate_derivative_network(adj_matrix, k):
    """Computes the derivative network of the graph

    Args:
        adj_matrix: adjacency matrix of the graph
        k: number of latent factors to keep

    Returns:
        derivative_network: the n2n derivative network of the graph
    """
    adj_matrix = mask_nan_entries_with_avg(adj_matrix)
    U, s, Vh = np.linalg.svd(adj_matrix, full_matrices=False)
    s = np.diag(s)
    derivative_network = 2 * U[:, k+1:] @ s[k+1:, k+1:] @ Vh[k+1:, :]
    return derivative_network

def predict_ratings():
    return None

def perturb_matrix(adj_matrix, num_perturbs, k):
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
        derivative_network = calculate_derivative_network(perturbed_adj_matrix, k)
        max_edge = find_max_edge(derivative_network)
        perturbed_adj_matrix[max_edge[0]][max_edge[1]] = np.nan
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
            if np.isnan(orig_adj_matrix[i, j]):
                error += (orig_svd[i, j] - perturbed_svd[i, j])**2
    error = error / omega_c
    error = np.sqrt(error)
    return error

def main(argv):
    if len(argv) < 1:
        print("Please mention a dataset (movielens or jester)")
        return

    k = 12
    num_perturbs = 20
    dataset_name = argv[0]
    if dataset_name == "lastfm":
        num_perturbs = 10
        k = 50
    
    adj_matrix = load_data(dataset_name)
    perturbed_matrix = perturb_matrix(adj_matrix, num_perturbs, k)

    omega_c = np.count_nonzero(np.isnan(adj_matrix))
    orig_svd = svd(adj_matrix, k)
    perturbed_svd = svd(perturbed_matrix, k)
    error = evaluate_error(orig_svd, perturbed_svd, adj_matrix, omega_c)
    print(error)
    
if __name__ == "__main__":
    main(sys.argv[1:])