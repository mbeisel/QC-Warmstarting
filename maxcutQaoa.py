import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, execute
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram
from helperFunctions import *

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
    shots = 5000
    QAOA = QAOACircuitGenerator.genQaoaMaxcutCircuit(Graph, input, approximation_List, p)
    QAOA_results = execute(QAOA, backend, shots=shots).result()
    return QAOA_results


def compute_costs(QAOA_results, G,inputCut = None, knownMaxCut = None, method = None, method_params =None, showHistogram=False):
    # Evaluate the data from the simulator
    counts = QAOA_results.get_counts()
    max_Cut_Probability = 0

    allCosts = np.array([cost_function_C(parseSolution(x), G) for x in list(counts.keys())])
    z = zip(list(counts.keys()), list(counts.values()), list(allCosts))
    z = list(z)

    # CLASSIC ENERGY CALCULATION
    #with offset
    # M1_sampled = (np.sum(np.array([allCosts[i] * z[i][1] for i in range(len(z))])) / np.sum(list(counts.values()))) - totalCost(G)
    #without offset

    if method == None:
        total_objective_value = (np.sum(np.array([allCosts[i] * z[i][1] for i in range(len(z))])) / np.sum(list(counts.values())))

    # ENERGY BASED ON ONLY BETTER RESULTS ONLY REWARD GOOD RESULTS
    elif inputCut and method.lower() == "greedy":
        n_samples = np.sum(list(counts.values()))
        if n_samples > 0:
            total_objective_value =  np.sum(np.array([allCosts[i] * z[i][1] if allCosts[i] > inputCut else 0 for i in range(len(z))])) / n_samples

    # ENERGY BASED ON ONLY BETTER RESULTS ONLY REWARD GOOD RESULTS - EXTRA REWARDS FOR BEST RESULT
    elif inputCut and method.lower() == "greedy_extra":
        n_samples = np.sum(list(counts.values()))
        if n_samples > 0:
            total_objective_value =  np.sum(np.array([(allCosts[i] * z[i][1] if allCosts[i] != np.max(allCosts) else (allCosts[i] * z[i][1])*10 ) if allCosts[i] > inputCut else 0 for i in range(len(z))]))/n_samples

    # Exclude initial cut from Energyfunction
    elif inputCut and method.lower() == "ee-i":
        total_objective_value =  np.sum(np.array([allCosts[i] * z[i][1] if allCosts[i] != inputCut else 0 for i in range(len(z))]))
        n_samples = np.sum(np.array([z[i][1] if allCosts[i] != inputCut else 0 for i in range(len(z))]))
        if n_samples > 0:
            total_objective_value = total_objective_value/n_samples

    # Differenz Methode
    elif inputCut and method.lower() == "diff":
        total_objective_value =  np.sum(np.array([(allCosts[i] - inputCut) * z[i][1] for i in range(len(z))]))
        n_samples = np.sum(list(counts.values()))
        if n_samples > 0:
            total_objective_value = total_objective_value/n_samples

    # CVar
    elif inputCut and method.lower() == "cvar":
        z.sort(key=takeThird, reverse=True)
        total_objective_value = 0
        alpha, *_ = method_params
        alpha *= np.ceil(np.sum(list(counts.values())))
        alphaRemaining = alpha
        for i in range(len(z)):
            if z[i][1] < alphaRemaining:
                total_objective_value += z[i][1] * z[i][2]
                alphaRemaining -= z[i][1]
            else:
                total_objective_value += alphaRemaining * z[i][2]
                break
        total_objective_value /= alpha

    # Gibbs
    elif inputCut and method.lower() == "gibbs":
        eta, *_ = method_params
        n_samples = np.sum(list(counts.values()))
        z = np.array(z, dtype=object)
        total_objective_value = np.log(np.sum((np.e ** (eta * z[:,2]))* z[:,1])/n_samples)


    best_sampled_cut_size = np.amax(allCosts)
    best_sampled_cut_string = parseSolution(z[np.where(allCosts == best_sampled_cut_size)[0][0]][0])

    n_samples = np.sum(list(counts.values()))
    better_cut_probability= 0
    if (inputCut and n_samples != 0):
        better_cut_probability = np.sum(np.array([z[i][1] if allCosts[i] > inputCut else 0 for i in range(len(z))])) / n_samples

    if (knownMaxCut):
        tupels = np.array(z)[np.where(allCosts == knownMaxCut)]
        max_Cut_Probability = np.sum([int(tuple[1])  for tuple in tupels])
        max_Cut_Probability = max_Cut_Probability/np.sum(list(counts.values()))

    if (showHistogram):
        plot_histogram(counts)
        plt.show()
        print("rank {}".format(np.where(allCosts == best_sampled_cut_size)[0][0]))
    # print("Max number of states: {} ".format(2 ** len(max_C[0])))
    # print("Number of achieved QAOA states: {} ".format(len(counts)))
    # print("Ratio of achieved states compared to max states {} ".format((len(counts) / (2 ** len(max_C[0])))*100))
    # print("Methode: {}".format(method))
    # print("Average: {}".format(M1_sampled))
    # print("Best Cut: {}".format(max_C[1]))
    # print("Best Cut State: {}".format(max_C[0]))

    return total_objective_value, best_sampled_cut_string, best_sampled_cut_size, max_Cut_Probability, better_cut_probability


def plotSolution(G, params, p):
    results = runQaoa(params, G, p)
    costs, solution, _, _, _ = compute_costs(results, G)
    GraphPlotter.plotGraph(G, solution)


def plotCircuit(G, approximation_List, params, p, backend=None):
    circuit = QAOACircuitGenerator.genQaoaMaxcutCircuit(G, params, approximation_List, p)

    circuit.draw(output='mpl')
    plt.show()

    if (backend):
        tcircuit = transpile(circuit, backend=backend)
        tcircuit.draw(output='mpl')
        plt.show()


def objectiveFunction(input, Graph, approximation_List, p, mixedOptimizerVars = None, inputCut = None, method = None,method_params =None, showHistogram=False, maxCut=None):
    if mixedOptimizerVars:
        input = mixedOptimizerVars + (list(input))
    results = runQaoa(input, Graph, approximation_List, p)
    costs, _, bestCut, maxCutChance, betterCutChance = compute_costs(results, Graph, showHistogram=showHistogram, method=method, method_params=method_params, inputCut=inputCut, knownMaxCut=maxCut)
    if method == "max":
        return -maxCutChance
    return -costs


def objectiveFunctionBest(input, Graph, approximation_List, p, knownMaxCut = None, inputCut = None, method = None,method_params =None, showHistogram=False):
    results = runQaoa(input, Graph, approximation_List, p)
    energy, _, bestCut, maxCutChance, betterCutChance = compute_costs(results, Graph, knownMaxCut=knownMaxCut, method=method, method_params=method_params, showHistogram=showHistogram, inputCut=inputCut)
    return energy, bestCut, maxCutChance, betterCutChance




