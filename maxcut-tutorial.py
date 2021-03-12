import numpy as np
import networkx as nx  # tool to handle general Graphs
from cvxgraphalgs.algorithms.max_cut import _solve_cut_vector_program

from graphGenerator import GraphGenerator, GraphPlotter
from QAOACircuitGenerator import QAOACircuitGenerator
import matplotlib.pyplot as plt 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from scipy.optimize import minimize

from qiskit import Aer, IBMQ, execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile, assemble
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
from qiskit.test.mock import FakeBoeblingen, FakeYorktown
import cvxgraphalgs as cvxgr
from numpy import dtype
from copy import deepcopy

# Compute the value of the cost function
def cost_function_C(x,G):
    
    E = G.edges()
    if( len(x) != len(G.nodes())):
        return np.nan
        
    C = 0;
    for index in E:
        e1 = index[0]
        e2 = index[1]
        
        w      = G[e1][e2]['weight']
        C = C + w*x[e1]*(1-x[e2]) + w*x[e2]*(1-x[e1])
        
    return C

def runQaoa(input, Graph, approximation_List, p):
    # run on local simulator
    backend = Aer.get_backend("qasm_simulator")
    shots = 2000
    QAOA = QAOACircuitGenerator.genQAOAcircuit(input, Graph,approximation_List, p)
    TQAOA = transpile(QAOA, backend)
    qobj = assemble(TQAOA)
    QAOA_results = execute(QAOA, backend, shots=shots).result()
    return QAOA_results

def compute_costs(QAOA_results, G):
    # Evaluate the data from the simulator
    counts = QAOA_results.get_counts()
    avr_C       = 0
    max_C       = [0,0]
    
    for sample in list(counts.keys()):
    
        # use sampled bit string x to compute C(x)
        x         = [int(num) for num in list(sample)]
        tmp_eng   = cost_function_C(x,G)
        
        # compute the expectation value and energy distribution
        avr_C     = avr_C    + counts[sample]*tmp_eng
        
        # save best bit string
        if( max_C[1] < tmp_eng):
            max_C[0] = sample
            max_C[1] = tmp_eng


    M1_sampled   = avr_C/np.sum(list(counts.values()))
    
    # print('The sampled mean value is M1_sampled = %.02f' % (M1_sampled))
    # print('The approximate solution is x* = %s with C(x*) = %d' % (max_C[0],max_C[1]))
    return M1_sampled, max_C[0], max_C[1]

def plotSolution(G, params, p):
    results = runQaoa(params, G, p)
    costs, solution, _ = compute_costs(results, G)
    GraphPlotter.plotGraph(G, solution)

def plotCircuit(G, approximation_List, params, p, backend = None):
    circuit = QAOACircuitGenerator.genQAOAcircuit(params, G,approximation_List, p)

    circuit.draw(output='mpl')
    plt.show()

    if(backend):
        tcircuit = transpile(circuit, backend=backend)
        tcircuit.draw(output='mpl')
        plt.show()

costs_history = []

def objectiveFunction(input, Graph, approximation_List, p):
    global costs_history
    results = runQaoa(input, Graph, approximation_List, p)
    costs, _, _ = compute_costs(results, Graph)
    costs_history.append(costs)
    return - costs

def objectiveFunctionBest(input, Graph, approximation_List, p):
    results = runQaoa(input, Graph, approximation_List, p)
    _, _, best = compute_costs(results, Graph)
    return best



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

def epsilonFunction(cutList, epsilon=0.25):
    # increase distance of continuous values from exact 0 and 1
    for i in range(len(cutList)):
        if(cutList[i] > 1-epsilon):
            cutList[i] = 1-epsilon
        if(cutList[i] < epsilon):
            cutList[i] = epsilon
    return cutList

def swapSign(list):
    return [ -i for i in list ]


def bestGWcuts(graph, n_GW_cuts, n_best, continuous=False, epsilon=0.25):
    # returns n_best best cuts out of n_GW_cuts to be computed
    if n_best > n_GW_cuts:
        raise Exception("n_best has to be less or equal to n_GW_cuts")

    GW_cuts = []
    for i in range(n_GW_cuts):

        if continuous:
            approximation_list = continuousGWsolve(graph)
        else:
            approximation = cvxgr.algorithms.goemans_williamson_weighted(graph)
            # compute binary representation of cut for discrete solution
            approximation_list = []
            for n in range(len(approximation.vertices)):
                if(n in approximation.left):
                    approximation_list.append(0)
                else:
                    approximation_list.append(1)

        GW_cuts.append([epsilonFunction(approximation_list, epsilon=epsilon), cost_function_C(approximation_list, graph)])

    GW_cuts = np.array(GW_cuts, dtype=object)
    GW_cuts = GW_cuts[GW_cuts[:,1].argsort()]
    GW_cuts = GW_cuts[n_GW_cuts-n_best:]
    return GW_cuts

# Gridsearch for p = 1
def gridSearch(objective_fun, Graph, approximation_List, p, step_size=0.2, show_plot=False):
    a_gamma         = np.arange(0, np.pi, step_size)
    a_beta          = np.arange(0, np.pi, step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma,a_beta)
    a_gamma, a_beta = a_gamma.flatten(), a_beta.flatten()

    F1 = np.array([objective_fun([a_gamma[i], a_beta[i]], Graph, approximation_List, p) for i in range(len(a_gamma))])

    # Grid search for the minimizing variables
    result = np.where(F1 == np.amin(F1))
    gamma, beta = a_gamma[result[0][0]], a_beta[result[0][0]]

    # Plot the expetation value F1
    if show_plot:
        fig = plt.figure()
        ax  = fig.gca(projection='3d')

        size = len(np.arange(0, np.pi, step_size))
        a_gamma, a_beta, F1 = a_gamma.reshape(size, size), a_beta.reshape(size, size), F1.reshape(size, size)
        surf = ax.plot_surface(a_gamma, a_beta, F1, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        ax.set_zlim(np.amin(F1)-1,np.amax(F1)+1)
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        plt.show()

    return np.array([gamma, beta]), np.amin(F1)

def compareWarmStartEnergy(graph, p_range):
    warm_means = []
    cold_means = []
    warm_dev = []
    cold_dev = []
    warm_max = []
    cold_max = []

    bestCuts = bestGWcuts(graph, 8, 5, continuous=False, epsilon=0)
    bestCuts = np.array([[epsilonFunction(cut[0], 0.25), cut[1]] for cut in deepcopy(bestCuts)], dtype=object)
    print(bestCuts)

    p_range = list(p_range)
    for p in p_range:
        warmstart = []
        coldstart = []

        optimizer_options = None #({"maxiter": 10})# to limit optimizer iterations
        for i in range(3,4):

            # bestCutsParams = [gridSearch(objectiveFunction, graph, bestCuts[i,0], p)[0] for i in range(len(bestCuts))]
            for j in range(1):
                params = np.random.default_rng().uniform(0, np.pi, size=2*p)
                params_warm_optimized = minimize(objectiveFunction, params, method="COBYLA", args=(graph, bestCuts[i,0],p), options=optimizer_options)
                # plotCircuit(graph, bestCuts[i,0], params_warm_optimized.x, p,)
                params_cold_optimized = minimize(objectiveFunction, params, method="COBYLA", args=(graph, None, p), options=optimizer_options)
                warmstart.append(objectiveFunctionBest(params_warm_optimized.x, graph, bestCuts[i,0], p))
                coldstart.append(objectiveFunctionBest(params_cold_optimized.x, graph, None, p))
            print("{:.2f}%".format(100*(i+1+5*p_range.index(p))/(len(p_range)*5)))

        warm_means.append(np.mean(warmstart))
        cold_means.append(np.mean(coldstart))
        warm_dev.append(np.std(warmstart))
        cold_dev.append(np.std(coldstart))
        warm_max.append(np.min(warmstart))
        cold_max.append(np.min(coldstart))
        print(warmstart)
        print(coldstart)

    print([warm_means, cold_means])
    print([warm_max, cold_max])
    plotline, capline, barlinecols = plt.errorbar(p_range, cold_means, cold_dev, linestyle="None", marker="x", color="b")
    [(bar.set_alpha(0.5), bar.set_label("coldstarted")) for bar in barlinecols]
    plotline, capline, barlinecols = plt.errorbar(p_range, warm_means, warm_dev, linestyle="None", marker="x", color="r")
    [(bar.set_alpha(0.5), bar.set_label("warmstarted")) for bar in barlinecols]
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("Energy"), plt.title("Warm-started QAOA comparison")
    plt.legend(loc="best")
    plt.show()

def compareEpsilon(graph, epsilon_range):
    warm_means = []
    warm_dev = []

    RawBestCuts = bestGWcuts(graph, 1, 1, continuous=False, epsilon=0) # get raw solutions using epsilon = 0
    print(RawBestCuts)
    p = 1

    epsilon_range = list(epsilon_range)
    for eps in epsilon_range:
        warmstart = []
        coldstart = []
        bestCuts = np.array([[epsilonFunction(cut[0], eps), cut[1]] for cut in deepcopy(RawBestCuts)], dtype=object)
        bestCutsParams = [gridSearch(objectiveFunction, graph, bestCuts[i,0], p)[0] for i in range(len(bestCuts))]
        optimizer_options = ({"rhobeg": 0.1, "disp": False})#, "maxiter": 10})# to limit optimizer iterations
        for i in range(-1,0):
            params = bestCutsParams[i]
            for j in range(5):
                #params = np.random.default_rng().normal(0, np.pi, size=2*p)
                params_warm_optimized = minimize(objectiveFunction, params, method="COBYLA", args=(graph, bestCuts[i,0], p), options=optimizer_options)
                warmstart.append(objectiveFunction(params_warm_optimized.x, graph, bestCuts[i,0], p))
                print("params optimized {} -> {}".format(params, params_warm_optimized.x))
            print("{:.2f}%".format(100*(i+1+5*epsilon_range.index(eps))/(len(epsilon_range)*5)))

        warm_means.append(np.mean(warmstart))
        warm_dev.append(np.std(warmstart))

    print(warmstart)
    print(warm_means)
    plotline, capline, barlinecols = plt.errorbar(epsilon_range, swapSign(warm_means), warm_dev, linestyle="None", marker="x", color="r")
    [(bar.set_alpha(0.5), bar.set_label("warmstarted")) for bar in barlinecols]
    plt.legend(loc="best"), plt.xlabel("epsilon"), plt.ylabel("Energy"), plt.title("Warm-started QAOA comparison")
    plt.show()

p = 1
params = np.random.default_rng().uniform(0, np.pi, size=2*p)
# graph = GraphGenerator.genButterflyGraph()
# graph = GraphGenerator.genGridGraph(4,4)
# graph = GraphGenerator.genFullyConnectedGraph(19, [-5,10])
# graph = GraphGenerator.genMustyGraph()
graph = GraphGenerator.genRandomGraph(22,177)
GraphPlotter.plotGraph(graph)

compareWarmStartEnergy(graph, [1,2])
# compareEpsilon(graph, np.arange(0.0,0.51,0.05))

# print(params.x)
# plotCircuit(graph, params.x, p, FakeYorktown())
# plotSolution(graph, params.x, p)
# plt.plot(costs_history)
# plt.show()