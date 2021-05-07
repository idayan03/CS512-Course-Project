import numpy as np
import networkx as nx
from copy import deepcopy

from n2n import svd, evaluate_error, perturb_matrix
from load_data import load_data, get_adj_matrix
from aurora_v2.utils import power_method_left, power_method_right, power_method_right_with_scaling

def comparison_error(adj_matrix, perturbed_matrix,k):
    """
    Putting functions together for evaluation 
    """
    omega_c = np.count_nonzero(np.isnan(adj_matrix))
    orig_svd = svd(adj_matrix, k)
    perturbed_svd = svd(perturbed_matrix, k)
    error = evaluate_error(orig_svd, perturbed_svd, adj_matrix, omega_c)
    return error

def get_graph(adj_matrix):
    """
    converts the adjacency matrix into networkx graph 
    Args:
        adj_matrix: data adjacency matrix
    Returns:
        graph with nodes same as adj_matrix indices
    """
    rows, cols = np.where(adj_matrix >= 0)
    edges = zip(rows.tolist(), cols.tolist())
    graph = nx.Graph()
    all_rows = range(0, adj_matrix.shape[0])
    for n in all_rows:
        graph.add_node(n)
    graph.add_edges_from(edges)
    return graph

def deg_centrailty_influence(adj_matrix, k):
    """
    calculates the influence based on degree centrality
    Args:
        adj_matrix: data adjacency matrix
        k: budget to pick top-k influentail edges
    Returns:
        S: list of top-k influential edges 
    """
    B = get_graph(adj_matrix)
    S = []
    while len(S) <= k:
        scores = nx.degree_centrality(B)
        for e1,e2 in B.edges():
            B[e1][e2]['influence'] = scores[e1]+scores[e2]
        max_e = [(e[0],e[1]) for e in sorted(list(B.edges(data=True)), 
                                            key=lambda x: x[2]['influence'], reverse=True)][:1][0]
        B.remove_edge(max_e[0], max_e[1])
        S.append(max_e)
    return S

def eigen_centrailty_influence(adj_matrix, k):
    """
    calculates the influence based on eignevector centrality
    Args:
        adj_matrix: data adjacency matrix
        k: budget to pick top-k influentail edges
    Returns:
        S: list of top-k influential edges 
    """
    B = get_graph(adj_matrix)
    S = []
    while len(S) <= k:
        scores = nx.eigenvector_centrality(B)
        for e1,e2 in B.edges():
            B[e1][e2]['influence'] = scores[e1]+scores[e2]
        max_e = [(e[0],e[1]) for e in sorted(list(B.edges(data=True)), 
                                            key=lambda x: x[2]['influence'], reverse=True)][:1][0]
        B.remove_edge(max_e[0], max_e[1])
        S.append(max_e)
    return S

def hits_influence(adj_matrix, k):
    """
    calculates the influence based on HITS
    Args:
        adj_matrix: data adjacency matrix
        k: budget to pick top-k influentail edges
    Returns:
        S: list of top-k influential edges 
    """
    B = get_graph(adj_matrix)
    S = []
    while len(S) <= k:
        scores = nx.hits(B)
        hub = scores[0]
        auth = scores[1]
    
        for e1,e2 in B.edges():
            B[e1][e2]['influence'] = hub[e1]+hub[e2] + auth[e1] + auth[e2]
    
        max_e = [(e[0],e[1]) for e in sorted(list(B.edges(data=True)), 
                                            key=lambda x: x[2]['influence'], reverse=True)][:1][0]
        B.remove_edge(max_e[0], max_e[1])
        S.append(max_e)
    return S



def get_r(graph, start_vector, max_iter, alpha, tol):
    r = deepcopy(start_vector)
    for num_iter in range(max_iter):
        xlast = r
        r = dict.fromkeys(xlast.keys(), 0)
        for n in r:
            for nbr in graph[n]:
                r[n] += alpha * xlast[nbr] * graph[n][nbr]['weight']
            r[n] += (1.0 - alpha) * start_vector[n]
        err = sum([abs(r[n] - xlast[n]) for n in r])
    return r

def aurora(adj_matrix, k):
    """
    calculates the influence based on AURORA_E for edges 
    Args:
        adj_matrix: data adjacency matrix
        k: budget to pick top-k influentail edges
    Returns:
        S: list of top-k influential edges 
    """

    alpha=0.85
    max_iter=100
    tol=1e-3
    B = get_graph(adj_matrix)
    N = len(B.nodes())
    for e in B.edges():
        B[e[0]][e[1]]['weight'] = 1
    directed = nx.is_directed(B)
    S = []

    start_vector = dict.fromkeys(B, 1.0 / N)
    while len(S)<= k:
        max_grad = float('-inf')
        e_max = (-1, -1, -1)
        max_iter = 100
        tol = 1e-3
        edges = set(B.edges(data='weight', default=1))
        r = get_r(B, start_vector, alpha=alpha, max_iter=max_iter, tol=tol)
        x = power_method_right_with_scaling(B, start_vector, alpha=alpha, max_iter=max_iter, tol=tol)
        for e in edges:
            u = e[0]
            v = e[1]
            if directed:
                gradient = r[v] * x[u]
            else:
                gradient = r[u] * x[v] + r[v] * x[u]
                if u == v:
                    gradient -= (r[v] * x[u])
            if gradient > max_grad:
                e_max = e
                max_grad = gradient
        B.remove_edge(e_max[0], e_max[1])
        S.append((e_max[1],e_max[0]))
    return S
def get_Ahat(adj_matrix,S):
    '''
    Args:
        adj_matrix: adjacency matrix
        S: selected top-k influential edges
    Returns:
        Ahat: the perturbed matrix from the comparison methods 
    '''
    Ahat = adj_matrix.copy()
    for e in S:
        Ahat[e[0]][e[1]] = np.nan
    return Ahat

def main():
    baseline_errors = {}
    k = 8
    budget = 4

    # adj_matrix = load_data("movielens")
    adj_matrix = load_data("jester")
    perturbed_matrix = perturb_matrix(adj_matrix, budget, k)
    error = comparison_error(adj_matrix,perturbed_matrix,k)
    baseline_errors['ours'] = error
    print('ours',error)
    # train_ratings_data, test_ratings_data = split_dataset(ratings_data, seed)

    # degree centrality
    S = deg_centrailty_influence(adj_matrix, budget)
    Ahat = get_Ahat(adj_matrix,S)
    error = comparison_error(adj_matrix,Ahat,k)
    baseline_errors['degree_centrality'] = error
    print("deg cent: ",error)
    
    # eigenvector centrality
    S = eigen_centrailty_influence(adj_matrix, budget)
    Ahat = get_Ahat(adj_matrix,S)
    error = comparison_error(adj_matrix,Ahat,k)
    baseline_errors['eign_centrality'] = error
    print("eigenvector cent: ", error)

    # HITS
    S = hits_influence(adj_matrix,budget)
    Ahat = get_Ahat(adj_matrix,S)
    error = comparison_error(adj_matrix,Ahat,k)
    baseline_errors['hits'] = error
    print("hits: ", error)

    # AUORORA 
    S = aurora(adj_matrix,budget)
    Ahat = get_Ahat(adj_matrix,S)
    error = comparison_error(adj_matrix,Ahat,k)
    baseline_errors['aurora'] = error
    print('aurora: ', error)
if __name__ == "__main__":
    main()