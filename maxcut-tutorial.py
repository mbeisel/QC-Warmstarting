import numpy as np
import networkx as nx  # tool to handle general Graphs
from cvxgraphalgs.algorithms.max_cut import _solve_cut_vector_program

from graphGenerator import GraphGenerator, GraphPlotter
from QAOACircuitGenerator import QAOACircuitGenerator
import matplotlib.pyplot as plt 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.optimize import minimize

from qiskit import Aer, IBMQ, execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile, assemble
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
from qiskit.test.mock import FakeBoeblingen, FakeYorktown
import cvxgraphalgs as cvxgr



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
    shots = 100
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
    return M1_sampled, max_C[0]

def plotSolution(G, params, p):
    results = runQaoa(params, G, p)
    costs, solution = compute_costs(results, G)
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
    costs, _ = compute_costs(results, Graph)
    costs_history.append(costs)
    return - costs


p = 1
params = np.random.default_rng().uniform(0, np.pi, size=2*p)
# graph = GraphGenerator.genButterflyGraph()
# graph = GraphGenerator.genGridGraph(4,4)
graph = GraphGenerator.genFullyConnectedGraph(10)
# graph = GraphGenerator.genMustyGraph()
# graph = GraphGenerator.genRandomGraph(12,20)
GraphPlotter.plotGraph(graph)

# for continous GW
# adjacency = nx.linalg.adjacency_matrix(graph)
# adjacency = adjacency.toarray()
# solution = _solve_cut_vector_program(adjacency)
# # solution = 1/np.linalg.norm(solution) * solution
#
# size = len(solution)
# partition = np.random.default_rng().uniform(size=size)
# partition_norm = np.linalg.norm(partition)
# partition = 1/partition_norm * partition
# projections = solution.T @ partition
# positive_projections = []
# for projection in range(len(projections)):
#     positive_projections.append((projections[projection]+1)/2 )

def epsilonFunction(cutList, epsilon):
    # approximation_list = positive_projections
    for i in range(len(cutList)):
        if(cutList[i] > 1-epsilon):
            cutList[i] = 1-epsilon
        if(cutList[i] < epsilon):
            cutList[i] = epsilon
    return cutList

GW_cuts = []
n_GW_cuts = 30
for i in range(n_GW_cuts):
    approximation = cvxgr.algorithms.goemans_williamson_weighted(graph)
    approximation_list = []
    for n in range(len(approximation.vertices)):
        if(n in approximation.left):
            approximation_list.append(0)
        else:
            approximation_list.append(1)
    GW_cuts.append([epsilonFunction(approximation_list, 0.25), cost_function_C(approximation_list, graph)])
print(GW_cuts)

NP_GW_cuts = np.array(GW_cuts)
NP_GW_cuts = NP_GW_cuts[NP_GW_cuts[:,1].argsort()]
NP_GW_cuts = NP_GW_cuts[n_GW_cuts-5:]
print(NP_GW_cuts)


warmstart = []
coldstart = []
# options=({"maxiter": 10}) to limit optimizer iterations
for i in range(5):
    for j in range(2):
        params = np.random.default_rng().uniform(0, np.pi, size=2*p)
        params_warm_optimized = minimize(objectiveFunction, params, method="COBYLA", args=(graph, NP_GW_cuts[i,0],p))
        params_cold_optimized = minimize(objectiveFunction, params, method="COBYLA", args=(graph, None , p))
        warmstart.append(objectiveFunction(params_warm_optimized.x, graph, NP_GW_cuts[i,0], p))
        coldstart.append(objectiveFunction(params_cold_optimized.x, graph, None, p))
    print(i)


print(warmstart)
print(np.mean(warmstart))
print(coldstart)
print(np.mean(coldstart))



# warmstart = []
# coldstart = []
# for x in range(100):
#     params = np.random.default_rng().uniform(0, np.pi, size=2*p)
#     warmstart.append(objectiveFunction(params, graph, approximation_list, p))
#     coldstart.append(objectiveFunction(params, graph, None, p))

# print(warmstart)
# print(np.mean(warmstart))
# print(coldstart)
# print(np.mean(coldstart))


# print(params.x)
# plotCircuit(graph, params.x, p, FakeYorktown())
# plotSolution(graph, params.x, p)
#plt.plot(costs_history)
#plt.show()