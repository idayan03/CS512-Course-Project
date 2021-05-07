import os
import numpy as np
import pandas as pd
import networkx as nx

def load_data(dataset_name):
    if dataset_name == "jester":
        return load_jester("datasets/jester.xlsx")
    elif dataset_name == "movielens":
        return load_movielens("datasets/ml-latest-small/ratings.csv")
    elif dataset_name == "lastfm":
        path = os.path.join("datasets", "lastfm.dat")
        return load_lastfm(path)

def load_movielens(path):
    """Loads the MovieLens data from path

    Args:
        path: the path of the MovieLens data

    Returns:
        data: a numpy array of the adjacency matrix of the graph
    """
    data = pd.read_csv(path)
    data.drop(columns=['timestamp'], inplace=True)  # Timestamp not needed
    return get_adj_matrix(data)

def load_jester(path):
    """Loads the Jester dataset from path

    Args:
        path: the path of the jester dataset
    
    Returns:
        adj_matrix: the adjacency matrix of the jester ratings
    """
    data = pd.read_excel(path, header=None)
    data.drop(data.columns[0], axis=1, inplace=True)    # Drop 1st column
    data.replace(to_replace=99, value=np.nan, inplace=True)    # 99 corresponds to no rating
    adj_matrix = np.array(data)
    return adj_matrix

def load_lastfm(path):
    data = pd.read_csv(path, names=['user', 'artist', 'feedback'], sep='\t')
    
    userList = data.iloc[:, 0].tolist()
    artistList = data.iloc[:, 1].tolist()
    feedbackList = data.iloc[:, 2].tolist()

    users = list(set(userList))
    artists = list(set(artistList))

    users_index = {users[i] : i for i in range(len(users))}
    pd_dict = {artist : [np.nan for i in range(len(users))] for artist in artists}

    for i in range(len(data)):
        user = userList[i]
        artist = artistList[i]
        feedback = feedbackList[i]
        pd_dict[artist][users_index[user]] = feedback

    X = pd.DataFrame(pd_dict)
    X.index = users
    return np.array(X)

def split_dataset(data, seed, test_ratio=0.2):
    """Splits the dataset into train and test datasets

    (Not used)

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

    return np.array(X)