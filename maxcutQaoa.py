import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, execute
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram

from QAOACircuitGenerator import QAOACircuitGenerator
from graphGenerator import GraphPlotter

# Compute the value of the cost function
def cost_function_C(x, G):
    n_vertices = G.shape[0]

    C = 0
    for i in range(n_vertices):
        for j in range(i):
            C += G[i,j] * (not x[i] == x[j])

    return C

def totalCost(G):
    n_vertices = G.shape[0]
    C_total = 0
    for i in range(n_vertices):
        for j in range(1,n_vertices):
            if i < j and G[i,j] != 0:
                w = G[i,j]
                C_total += w
    return C_total/2


def runQaoa(input, Graph, approximation_List, p):
    # run on local simulator
    backend = Aer.get_backend("qasm_simulator")
    shots = 2000
    QAOA = QAOACircuitGenerator.genQaoaMaxcutCircuit(Graph, input, approximation_List, p)
    TQAOA = transpile(QAOA, backend)
    # qobj = assemble(TQAOA)
    QAOA_results = execute(QAOA, backend, shots=shots).result()
    return QAOA_results


def compute_costs(QAOA_results, G,inputCut = None, knownMaxCut = None, showHistogram=False):
    # Evaluate the data from the simulator
    counts = QAOA_results.get_counts()
    max_C = [0, 0, 0]
    max_Cut_Probability = 0

    z = zip(list(counts.keys()), list(counts.values()))
    z = list(z)


    def takeFirst(elem):
        return elem[0]
    def takeSecond(elem):
        return elem[1]
    def parseSolution(sol):
        return [int(i) for i in sol]

    z.sort(key=takeSecond, reverse=True)

    allCosts = np.array([cost_function_C(parseSolution(x), G) for x, _ in z])
    allCostsWeightedByNumberOfOccurances = np.array([allCosts[i] * z[i][1] for i in range(len(z))])


    # CLASSIC ENERGY CALCULATION
    #with offset
    M1_sampled = (   np.sum(allCostsWeightedByNumberOfOccurances) / np.sum(list(counts.values()))  ) - totalCost(G)
    #without offset
    # M1_sampled = (   np.sum(allCostsWeightedByNumberOfOccurances) / np.sum(list(counts.values()))  )

    # ENERGY BASED ON ONLY BETTER RESULTS ONLY REWARD GOOD RESULTS
    # if inputCut:
    #     M1_sampled =  np.sum(np.array([allCosts[i] * z[i][1] if allCosts[i] > inputCut else 0 for i in range(len(z))]))

    # ENERGY BASED ON ONLY BETTER RESULTS ONLY REWARD GOOD RESULTS PUNISH BAD ONES
    if inputCut:
        M1_sampled =  np.sum(np.array([allCosts[i] * z[i][1] if allCosts[i] > inputCut else -(allCosts[i] * z[i][1])/50 for i in range(len(z))]))

    print(M1_sampled)
    max_C[1] = np.amax(allCosts)
    max_C[0] = parseSolution(z[np.where(allCosts == max_C[1])[0][0]][0])

    # only take most common value as solution cut
    # max_C[1] = allCosts[0]
    # max_C[0] = bitarray.bitarray(z[0][1])

    if (knownMaxCut):
        tupels = np.array(z)[np.where(allCosts == knownMaxCut)]
        # print(tupels)
        max_Cut_Probability = np.sum([int(tuple[1])  for tuple in tupels])
        # print(max_Cut_Probability)
        max_Cut_Probability = max_Cut_Probability/np.sum(list(counts.values()))
        # print(max_Cut_Probability)

    if (showHistogram):
        plot_histogram(counts)
        plt.show()
        print("rank {}".format(np.where(allCosts == max_C[1])[0][0]))
    # print("Max number of states: {} ".format(2 ** len(max_C[0])))
    # print("Number of achieved QAOA states: {} ".format(len(counts)))
    # print("Ratio of achieved states compared to max states {} ".format((len(counts) / (2 ** len(max_C[0])))*100))
    # print("Average: {}".format(M1_sampled))
    # print("Best Cut: {}".format(max_C[1]))
    # print("Best Cut State: {}".format(max_C[0]))

    return M1_sampled, max_C[0], max_C[1], max_Cut_Probability


def plotSolution(G, params, p):
    results = runQaoa(params, G, p)
    costs, solution, _, _ = compute_costs(results, G)
    GraphPlotter.plotGraph(G, solution)


def plotCircuit(G, approximation_List, params, p, backend=None):
    circuit = QAOACircuitGenerator.genQaoaMaxcutCircuit(G, params, approximation_List, p)

    circuit.draw(output='mpl')
    plt.show()

    if (backend):
        tcircuit = transpile(circuit, backend=backend)
        tcircuit.draw(output='mpl')
        plt.show()


def objectiveFunction(input, Graph, approximation_List, p, mixedOptimizerVars = None, inputCut = None, showHistogram=False):
    if mixedOptimizerVars:
        input = mixedOptimizerVars + (list(input))
    results = runQaoa(input, Graph, approximation_List, p)
    costs, _, _, _ = compute_costs(results, Graph, showHistogram=showHistogram, inputCut=inputCut)
    return - costs


def objectiveFunctionBest(input, Graph, approximation_List, p, knownMaxCut = None, inputCut = None, showHistogram=False):
    results = runQaoa(input, Graph, approximation_List, p)
    energy, _, bestCut, maxCutChance = compute_costs(results, Graph, knownMaxCut=knownMaxCut, showHistogram=showHistogram, inputCut=inputCut)
    return energy, bestCut, maxCutChance




