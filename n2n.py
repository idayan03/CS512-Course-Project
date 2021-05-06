import random
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split

def load_data(path):
    """Loads the MovieLens data from path

    Args:
        path: the path of the MovieLens data

    Returns:
        data: a pandas DataFrame of the data
        users: a numpy array of the users
        movies: a numpy array of the movies
    """
    data = pd.read_csv(path)
    data.drop(columns=['timestamp'], inplace=True)  # Timestamp not needed
    users = data["userId"].unique()
    movies = data["movieId"].unique()
    return data, users, movies

def split_dataset(data, seed, test_ratio=0.2):
    """Splits the dataset into train and test datasets

    Args:
        data: a pandas DataFrame of the data
        test_ratio: the ratio of the test dataset (Default=0.2)

    Returns:
        train_ratings_data: a pandas DataFrame of the train data
        test_ratings_data: a pandas DataFrame of the test data
    """
    train_ratings_data, test_ratings_data = train_test_split(data, test_size=test_ratio, shuffle=True, random_state=seed)
    return train_ratings_data, test_ratings_data

def get_adj_matrix(data, formatizer = {'user':0, 'movie':1, 'rating':2}):
    """Returns the adjacency matrix of the data

    Args:
        data: a pandas DataFrame of the data

    Returns:
        X: a numpy array of the adjacency matrix
        users_index:
        movies_index:
    """
    userField = formatizer['user']
    movieField = formatizer['movie']
    ratingField = formatizer['rating']

    userList = data.iloc[:, userField].tolist()
    movieList = data.iloc[:, movieField].tolist()
    ratingList = data.iloc[:, ratingField].tolist()

    users = list(set(data.iloc[:, userField]))
    movies = list(set(data.iloc[:, movieField]))

    users_index = {users[i] : i for i in range(len(users))}
    pd_dict = {movie : [np.nan for i in range(len(users))] for movie in movies}

    for i in range(len(data)):
        user = userList[i]
        movie = movieList[i]
        rating = ratingList[i]
        pd_dict[movie][users_index[user]] = rating
    
    X = pd.DataFrame(pd_dict)
    X.index = users

    moviecols = list(X.columns)
    movies_index = {moviecols[i]:i for i in range(len(moviecols))}

    return np.array(X), users_index, movies_index

def mask_nan_entries_with_avg(adj_matrix):
    # the nan entries are masked
    mask = np.isnan(adj_matrix)
    masked_arr = np.ma.masked_array(adj_matrix, mask)
    movie_means = np.mean(masked_arr, axis=0)

    # nan entries will replaced by the average rating for each item
    adj_matrix = masked_arr.filled(movie_means)
    return adj_matrix

def svd(adj_matrix, k):
    """Computes the svd of adj_matrix

    Args:
        adj_matrix:
        k: the number of latent features to keep

    Returns:

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

def find_max_edge(derivative_network):
    """Finds the maximumum user-movie relationship (rating) of the derivative network
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

def main():
    k = 8
    seed = random.randint(0, 50)
    ratings_data, users, movies = load_data("ml-latest-small/ratings.csv")
    # train_ratings_data, test_ratings_data = split_dataset(ratings_data, seed)
    adj_matrix, users_index, movies_index = get_adj_matrix(ratings_data)
    perturbed_matrix = perturb_matrix(adj_matrix, 20, k)

    omega_c = np.count_nonzero(np.isnan(adj_matrix))
    orig_svd = svd(adj_matrix, k)
    perturbed_svd = svd(perturbed_matrix, k)
    error = evaluate_error(orig_svd, perturbed_svd, adj_matrix, omega_c)
    print(error)
    
if __name__ == "__main__":
    main()