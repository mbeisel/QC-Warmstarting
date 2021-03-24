import numpy as np

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
from numpy import dtype
import time
from datetime import datetime


# Compute the value of the cost function
def cost_function_C(x, G):
    n_vertices = G.shape[0]

    C = 0;
    C_total = 0
    for i in range(n_vertices):
        for j in range(1,n_vertices):
            if i < j and graph[i,j] != 0:
                w = graph[i,j]
                if(x[i] != x[j]):
                    C += w
                C_total += w
                # C += w * x[i] * (1 - x[j]) + w * x[j] * (1 - x[i])
    return C

def totalCost(G):
    n_vertices = G.shape[0]
    C_total = 0
    for i in range(n_vertices):
        for j in range(1,n_vertices):
            if i < j and graph[i,j] != 0:
                w = graph[i,j]
                C_total += w
    return C_total/2


def runQaoa(input, Graph, approximation_List, p):
    # run on local simulator
    backend = Aer.get_backend("qasm_simulator")
    shots = 2000
    QAOA = QAOACircuitGenerator.genQaoaMaxcutCircuit(Graph, input, approximation_List, p)
    TQAOA = transpile(QAOA, backend)
    qobj = assemble(TQAOA)
    QAOA_results = execute(QAOA, backend, shots=shots).result()
    return QAOA_results


def compute_costs(QAOA_results, G, knownMaxCut = None, showHistogram=False):
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

    M1_sampled = (   np.sum(allCostsWeightedByNumberOfOccurances) / np.sum(list(counts.values()))  ) - totalCost(G)
    max_C[1] = np.amax(allCosts)
    max_C[0] = parseSolution(z[np.where(allCosts == max_C[1])[0][0]][0])

    # only take most common value as solution cut
    # max_C[1] = allCosts[0]
    # max_C[0] = bitarray.bitarray(z[0][1])

    if (knownMaxCut):
        tupels = np.array(z)[np.where(allCosts == knownMaxCut)]
        print(tupels)
        max_Cut_Probability = np.sum([int(tuple[1])  for tuple in tupels])
        print(max_Cut_Probability)
        max_Cut_Probability = max_Cut_Probability/np.sum(list(counts.values()))
        print(max_Cut_Probability)

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


def objectiveFunction(input, Graph, approximation_List, p, showHistogram=False):
    results = runQaoa(input, Graph, approximation_List, p)
    costs, _, _, _ = compute_costs(results, Graph, showHistogram)
    return - costs


def objectiveFunctionBest(input, Graph, approximation_List, p, knownMaxCut = None, showHistogram=False):
    results = runQaoa(input, Graph, approximation_List, p)
    energy, _, bestCut, maxCutChance = compute_costs(results, Graph, knownMaxCut=knownMaxCut, showHistogram=showHistogram)
    return energy, bestCut, maxCutChance


def compareOptimizerEnergy(graph, p_range, optimizers):
    bestCuts = bestGWcuts(graph, 8, 5, continuous=False, epsilon=0)
    bestCuts = np.array([[epsilonFunction(cut[0], 0.25), cut[1]] for cut in deepcopy(bestCuts)], dtype=object)
    print(bestCuts)

    p_range = list(p_range)
    optimizers_p_cut_warm = [[] for j in range(len(optimizers))]
    optimizers_p_cut_std_warm = [[] for j in range(len(optimizers))]
    optimizers_p_energy_warm = [[] for j in range(len(optimizers))]
    optimizers_p_energy_std_warm = [[] for j in range(len(optimizers))]
    optimizers_p_runtime = [[] for j in range(len(optimizers))]

    for p in p_range:
        warmstart = []
        coldstart = []

        optimizer_options = None  # ({"maxiter": 10})# to limit optimizer iterations
        # optimizer_options = {"rhobeg":1.57}
        # take 3rd best cut
        for i in range(2, 3):
            params = [0, np.pi / 2] * p
            # params=  np.zeros(2*p)  #
            params = np.random.default_rng().uniform(0, np.pi, size=2 * p)
            print(params)
            for optimizer in range(len(optimizers)):
                resultCuts = []
                resultEnergy = []
                times = []
                for j in range(1):
                    t1 = time.time()
                    params_warm_optimized = minimize(objectiveFunction, params, method=optimizers[optimizer],
                                                     args=(graph, bestCuts[i, 0], p), options=optimizer_options)
                    energy, cut, _ = objectiveFunctionBest(params_warm_optimized.x, graph, bestCuts[i, 0], p,
                                                        showHistogram=False)
                    resultCuts.append(energy)
                    resultEnergy.append(-cut)
                    t2 = time.time()
                    times.append(t2 - t1)

                    # params_cold_optimized = minimize(objectiveFunction, params, method=optimizers[optimizer], args=(graph, None, p), options=optimizer_options)
                    # coldstart.append(objectiveFunctionBest(params_cold_optimized.x, graph, None, p))
                    # optimizers_p_values_cold[optimizer].append(objectiveFunctionBest(params_cold_optimized.x, graph, None, p))

                optimizers_p_cut_warm[optimizer].append(np.mean(resultCuts))
                optimizers_p_cut_std_warm[optimizer].append(np.std(resultCuts))
                optimizers_p_energy_warm[optimizer].append(np.mean(resultEnergy))
                optimizers_p_energy_std_warm[optimizer].append(np.std(resultEnergy))
                optimizers_p_runtime[optimizer].append(np.mean(times))
                print(optimizers_p_cut_warm)

        # with open('output.txt', 'w') as f:
        #     print(warmstart, file=f)
        #     print(coldstart, file=f)

    print(optimizers_p_cut_warm)
    for optimizer in range(len(optimizers)):
        print(optimizers[optimizer] + " Cut" + str(optimizers_p_cut_warm[optimizer]))
        print(optimizers[optimizer] + " Energy" + str(optimizers_p_energy_warm[optimizer]))
        # with open('output.txt', 'w') as f:
        #     print(optimizers[optimizer] + str(optimizers_p_values_warm)[optimizer], file=f)

    # warmstartgraph
    for optimizer in range(len(optimizers)):
        plotline, capline, barlinecols = plt.errorbar(p_range, optimizers_p_cut_warm[optimizer],
                                                      optimizers_p_cut_std_warm[optimizer], marker="x",
                                                      label=optimizers[optimizer] + "cut")
        [(bar.set_alpha(0.5)) for bar in barlinecols]
        plotline, capline, barlinecols = plt.errorbar(p_range, optimizers_p_energy_warm[optimizer],
                                                      optimizers_p_energy_std_warm[optimizer], marker="x",
                                                      label=optimizers[optimizer] + "energy")
    plt.plot([np.min(p_range), np.max(p_range)], [bestCuts[2, 1], bestCuts[2, 1]], linestyle="dashed",
             label="used GW-Cut")
    plt.plot([np.min(p_range), np.max(p_range)], [bestCuts[-1, 1], bestCuts[-1, 1]], linestyle="dashed",
             label="best GW-Cut")
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("Cutsize"), plt.title("Optimizer comparison warmstart")
    plt.savefig("results/optimizers_p_values_warm-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.show()
    plt.close()

    # coldstartgraph
    # for optimizer in range(len(optimizers)):
    #     plt.errorbar(p_range, optimizers_p_values_cold[optimizer], marker="x", label=optimizers[optimizer])
    # plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("Energy"), plt.title("Optimizer comparison coldstart")
    # plt.savefig('optimizers_p_values_cold.png')
    # plt.show()
    # plt.close()

    # runtimegraph
    for optimizer in range(len(optimizers)):
        plt.errorbar(p_range, optimizers_p_runtime[optimizer], marker="x", label=optimizers[optimizer])
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("Runtime in s"), plt.title(
        "Optimizer runtime comparison (warmstart)")
    plt.savefig("results/optimizer_p_energy-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.show()
    plt.close()


def compareWarmStartEnergy(graph, p_range, initialCut = None, knownMaxCut = None):
    warm_means = []
    cold_means = []
    warm_value_list = []
    cold_value_list = []
    warm_max = []
    cold_max = []
    warm_MaxCutProb = []
    cold_MaxCutProb = []
    warm_MaxCutProb_Values = []
    cold_MaxCutProb_Values = []

    bestCuts = bestGWcuts(graph, 8, 5, continuous=False, epsilon=0)
    bestCuts = np.array([[epsilonFunction(cut[0], epsilon=0.25), cut[1]] for cut in deepcopy(bestCuts)], dtype=object)
    if not knownMaxCut:
        knownMaxCut = bestCuts[len(bestCuts)-1,1]
    print("knownmaxcut {}".format(knownMaxCut))
    print(bestCuts)

    p_range = list(p_range)
    for p in p_range:
        warmstart = []
        coldstart = []
        warmstartMaxCutProb = []
        coldstartMaxCutProb = []

        optimizer_options = None  # ({"maxiter": 10})# to limit optimizer iterations
        for i in range(2, 3):

            # bestCutsParams = [gridSearch(objectiveFunction, graph, bestCuts[i,0], p)[0] for i in range(len(bestCuts))]
            for j in range(3):
                bestCut = bestCuts[i,0]
                if(initialCut):
                    bestCut = epsilonFunction(initialCut[0], epsilon=0.25)
                print(bestCut)
                params = np.zeros(2 * p)  # np.random.default_rng().uniform(0, np.pi, size=2*p)
                params_warm_optimized = minimize(objectiveFunction, params, method="COBYLA",
                                                 args=(graph, bestCut, p), options=optimizer_options)
                # plotCircuit(graph, bestCuts[i,0], params_warm_optimized.x, p,)
                params_cold_optimized = minimize(objectiveFunction, params, method="COBYLA", args=(graph, None, p),
                                                 options=optimizer_options)
                energyWarm, cutWarm, maxCutChanceWarm = objectiveFunctionBest(params_warm_optimized.x, graph, bestCut, p,
                                                                              knownMaxCut= knownMaxCut,
                                                                              showHistogram=False)
                warmstart.append(energyWarm)
                warmstartMaxCutProb.append(maxCutChanceWarm)
                print("maxcutchance {}".format(maxCutChanceWarm))
                energyCold, cutCold, maxCutChanceCold = objectiveFunctionBest(params_cold_optimized.x, graph, None, p,
                                                                              knownMaxCut= knownMaxCut,
                                                                              showHistogram=False)
                print("maxcutchancecold {}".format(maxCutChanceCold))
                coldstartMaxCutProb.append(maxCutChanceCold)
                coldstart.append(energyCold)
            print("{:.2f}%".format(100 * (i + 1 + 5 * p_range.index(p)) / (len(p_range) * 5)))

        warm_MaxCutProb.append(np.mean(warmstartMaxCutProb)*100)
        cold_MaxCutProb.append(np.mean(coldstartMaxCutProb)*100)
        warm_MaxCutProb_Values.append([[p for i in range(len(warmstartMaxCutProb))], np.array(warmstartMaxCutProb)*100])
        cold_MaxCutProb_Values.append([[p for i in range(len(coldstartMaxCutProb))], np.array(coldstartMaxCutProb)*100])
        warm_means.append(np.mean(warmstart))
        cold_means.append(np.mean(coldstart))
        warm_value_list.append([[p for i in range(len(warmstart))], warmstart])
        cold_value_list.append([[p for i in range(len(coldstart))], coldstart])
        warm_max.append(np.min(warmstart))
        cold_max.append(np.min(coldstart))
        print(warmstart)
        print(coldstart)

    print([warm_means, cold_means])
    print([warm_MaxCutProb, cold_MaxCutProb])
    # print([warm_max, cold_max])


    #energygraph
    warm_value_list = np.array(warm_value_list)
    cold_value_list = np.array(cold_value_list)
    plt.scatter(warm_value_list[:,0], warm_value_list[:,1], marker=".", color='red', label="warmstarted", alpha=.4)
    plt.scatter(cold_value_list[:,0], cold_value_list[:,1], marker=".", color='blue', label="coldstarted", alpha=.4)
    plt.scatter(p_range, warm_means, linestyle="None", marker="x", color="r", label="mean cut", alpha=.75)
    plt.scatter(p_range, cold_means, linestyle="None", marker="x", color="b", label="mean cut", alpha=.75)
    usedCut = bestCuts[2,1]
    offset = totalCost(graph)
    if(initialCut):
        usedCut = initialCut[1]
    plt.plot([np.min(p_range), np.max(p_range)], [usedCut -offset, usedCut-offset], linestyle="dashed",
             label="used GW-Cut")
    plt.plot([np.min(p_range), np.max(p_range)], [bestCuts[-1, 1]-offset, bestCuts[-1, 1]-offset], linestyle="dashed",
         label="best GW-Cut")
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("Energy"), plt.title("Warm-started QAOA comparison")
    plt.savefig("results/warmstartEnergy-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.show()

    #probabilitygraph
    warm_MaxCutProb_Values = np.array(warm_MaxCutProb_Values)
    cold_MaxCutProb_Values = np.array(cold_MaxCutProb_Values)
    plt.scatter(warm_MaxCutProb_Values[:,0], warm_MaxCutProb_Values[:,1], marker=".", color='red', label="warmstarted", alpha=.4)
    plt.scatter(cold_MaxCutProb_Values[:,0], cold_MaxCutProb_Values[:,1], marker=".", color='blue', label="coldstarted", alpha=.4)
    plt.scatter(p_range, warm_MaxCutProb, linestyle="None", marker="x", color="r", label="warm mean", alpha=.75)
    plt.scatter(p_range, cold_MaxCutProb, linestyle="None", marker="x", color="b", label="cold mean", alpha=.75)
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("MaxCut Probabilityin %"), plt.title(
        "MaxCut Probability")
    plt.savefig("results/warmstartEnergy-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.show()


# graph = GraphGenerator.genButterflyGraph()
# graph = GraphGenerator.genGridGraph(4,4)
# graph = GraphGenerator.genFullyConnectedGraph(5)
# graph = GraphGenerator.genMustyGraph()
# graph = GraphGenerator.genRandomGraph(5,6)
graph = GraphGenerator.genWarmstartPaperGraph()
# GraphPlotter.plotGraph(nx.Graph(graph))

# compareWarmStartEnergy(graph, [1,2,3 ], initialCut = [[0,0,1,1,1,1], 23], knownMaxCut = 27)
# compareOptimizerEnergy(graph, [1], ["Cobyla", "TNC"])  #TNC
# compareOptimizerEnergy(graph, [1,2], ["Cobyla", "Powell"])

# GraphPlotter.plotGraph(graph, fname="results/graph-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")

# print(params.x)
# plotCircuit(graph, params.x, p, FakeYorktown())
# plotSolution(graph, params.x, p)
# plt.plot(costs_history)
# plt.show()
