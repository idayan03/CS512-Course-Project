import numpy as np
import pandas as pd

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

    return np.array(X), users_index, movies_index