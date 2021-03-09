import numpy as np
import networkx as nx  # tool to handle general Graphs 
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

def runQaoa(input, Graph, p):        
    # run on local simulator
    backend = Aer.get_backend("qasm_simulator")
    shots = 100
    QAOA = QAOACircuitGenerator.genQAOAcircuit(input, Graph, p)
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
    
    print('The sampled mean value is M1_sampled = %.02f' % (M1_sampled))
    print('The approximate solution is x* = %s with C(x*) = %d' % (max_C[0],max_C[1]))
    return M1_sampled, max_C[0]

def plotSolution(G, params, p):
    results = runQaoa(params, G, p)
    costs, solution = compute_costs(results, G)
    GraphPlotter.plotGraph(G, solution)

def plotCircuit(G, params, p):
    circuit = QAOACircuitGenerator.genQAOAcircuit(params, G, p)
    circuit.draw(output='mpl')
    plt.show()

costs_history = []
def objectiveFunction(input, Graph, p):
    global costs_history
    results = runQaoa(input, Graph, p)
    costs, _ = compute_costs(results, Graph)
    costs_history.append(costs)
    return - costs

p = 1
params = np.random.default_rng().uniform(0, np.pi, size=2*p)
Graph = GraphGenerator.genButterflyGraph()
#Graph = GraphGenerator.genGridGraph()
#Graph = GraphGenerator.genMustyGraph()
#GraphPlotter.plotGraph(Graph)
params = minimize(objectiveFunction, params, method="COBYLA", args=(Graph, p))
print(params.x)
#plotCircuit(Graph, params.x, p)
plotSolution(Graph, params.x, p)
#plt.plot(costs_history)
#plt.show()