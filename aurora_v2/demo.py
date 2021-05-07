from scipy.sparse.linalg import eigs
import load_data as data
from utils import *
from aurora_e import *
from aurora_n import *
from aurora_s import *


if __name__ == '__main__':
    G = data.grqc()  # input graph
    query = None  # a list of indices for query node, if global, set it to None
    k = 10  # budget size


    max_iter = 50  # maximum number of iteration
    tol = 1e-3  # tolerance for error check in power method

    print("k =", k)
    N = G.number_of_nodes()
    A = nx.to_scipy_sparse_matrix(G, dtype='float', format='csc')
    eigval, _ = eigs(A, k=1)
    alpha = 0.5 / abs(eigval[0])
    alpha = alpha.real

    if query is None:
        start_vector = dict.fromkeys(G, 1.0 / N)
    else:
        start_vector = dict.fromkeys(G, 0.0)
        length = len(query)
        for i in query:
            start_vector[i] = 1.0 / length

    r = power_method_left(G, start_vector, alpha=alpha)

    edges = aurora_e(G, start_vector, alpha=alpha, budget=k, max_iter=max_iter, tol=tol)
    print(edges)
    goodness = evaluate(G, edges, start_vector, r, alpha=alpha, element='edge')
    print("AURORA-E:", goodness)

    nodes = aurora_n(G, start_vector, alpha=alpha, budget=k, query=query, max_iter=max_iter, tol=tol)
    print(nodes)
    goodness = evaluate(G, nodes, start_vector, r, alpha=alpha, element='node')
    print("AURORA-N:", goodness)

    subgraph = aurora_s(G, start_vector, alpha=alpha, budget=k, max_iter=max_iter, tol=tol)
    print(subgraph)
    goodness = evaluate(G, subgraph, start_vector, r, alpha=alpha, element='subgraph')
    print("AURORA-S:", goodness)
