import networkx as nx
import cvxgraphalgs as cvxgr
from cvxgraphalgs.algorithms.max_cut import _solve_cut_vector_program
from helperFunctions import epsilonFunction
from copy import deepcopy
import numpy as np

def bestGWcuts(graph, n_GW_cuts, n_best, cost_fun=None, continuous=False, epsilon=0.25, allow_duplicates=False):
    # returns n_best best cuts out of n_GW_cuts to be computed
    if n_best > n_GW_cuts:
        raise Exception("n_best has to be less or equal to n_GW_cuts")

    GW_cuts = []
    for i in range(n_GW_cuts):

        if continuous:
            approximation_list = continuousGWsolve(graph)
        else:
            approximation = cvxgr.algorithms.goemans_williamson_weighted(nx.Graph(graph))
            # compute binary representation of cut for discrete solution
            approximation_list = []
            for n in range(len(approximation.vertices)):
                if (n in approximation.left):
                    approximation_list.append(0)
                else:
                    approximation_list.append(1)

        epsilonizedCut = epsilonFunction(deepcopy(approximation_list), epsilon=epsilon) if epsilon else approximation_list
        if allow_duplicates or not GW_cuts or not (epsilonizedCut in list(np.array(GW_cuts, dtype=object)[:,0])):
            GW_cuts.append([epsilonizedCut, cost_fun(approximation_list, graph) if cost_fun else 0])

    GW_cuts = np.array(GW_cuts, dtype=object)
    GW_cuts = GW_cuts[GW_cuts[:, 1].argsort()]
    GW_cuts = GW_cuts[n_GW_cuts - n_best:]
    return GW_cuts

def continuousGWsolve(graph):
    # compute continuous valued, [0,1]-normalized GW solution
    adjacency = nx.linalg.adjacency_matrix(graph)
    adjacency = adjacency.toarray()
    solution = _solve_cut_vector_program(adjacency)

    size = len(solution)
    partition = np.random.default_rng().uniform(size=size)
    partition_norm = np.linalg.norm(partition)
    partition = 1 / partition_norm * partition
    projections = solution.T @ partition

    # normalize [-1,1] -> [0,1]
    positive_projections = (projections + 1) / 2
    return list(positive_projections)
