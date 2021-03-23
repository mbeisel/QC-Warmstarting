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
import bitarray
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

    # qubitOp, offset = max_cut.get_operator(G)
    # print('Offset:', offset)
    # print('Ising Hamiltonian:')
    # print(qubitOp.print_details())
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
    QAOA = QAOACircuitGenerator.genQAOAcircuit(input, Graph, approximation_List, p)
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

    z.sort(key=takeSecond, reverse=True)

    allCosts = np.array([cost_function_C(bitarray.bitarray(x), G) for x, _ in z])
    allCostsWeightedByNumberOfOccurances = np.array([allCosts[i] * z[i][1] for i in range(len(z))])

    M1_sampled = (   np.sum(allCostsWeightedByNumberOfOccurances) / np.sum(list(counts.values()))  ) - totalCost(G)
    max_C[1] = np.amax(allCosts)
    max_C[0] = bitarray.bitarray(z[np.where(allCosts == max_C[1])[0][0]][0])

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
    circuit = QAOACircuitGenerator.genQAOAcircuit(params, G, approximation_List, p)

    circuit.draw(output='mpl')
    plt.show()

    if (backend):
        tcircuit = transpile(circuit, backend=backend)
        tcircuit.draw(output='mpl')
        plt.show()


costs_history = []


def objectiveFunction(input, Graph, approximation_List, p, showHistogram=False):
    global costs_history
    results = runQaoa(input, Graph, approximation_List, p)
    costs, _, _, _ = compute_costs(results, Graph, showHistogram)
    costs_history.append(costs)
    return - costs


def objectiveFunctionBest(input, Graph, approximation_List, p, knownMaxCut = None, showHistogram=False):
    results = runQaoa(input, Graph, approximation_List, p)
    energy, _, bestCut, maxCutChance = compute_costs(results, Graph, knownMaxCut=knownMaxCut, showHistogram=showHistogram)
    return energy, bestCut, maxCutChance


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
        if (cutList[i] > 1 - epsilon):
            cutList[i] = 1 - epsilon
        if (cutList[i] < epsilon):
            cutList[i] = epsilon
    return cutList


def swapSign(list):
    return [-i for i in list]


def bestGWcuts(graph, n_GW_cuts, n_best, continuous=False, epsilon=0.25):
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

        GW_cuts.append(
            [epsilonFunction(deepcopy(approximation_list), epsilon=epsilon), cost_function_C(approximation_list, graph)])

    GW_cuts = np.array(GW_cuts, dtype=object)
    GW_cuts = GW_cuts[GW_cuts[:, 1].argsort()]
    GW_cuts = GW_cuts[n_GW_cuts - n_best:]
    return GW_cuts


# Gridsearch for p = 1
def gridSearch(objective_fun, Graph, approximation_List, p, step_size=0.2, plot=False, fname=None):
    a_gamma = np.arange(0, np.pi, step_size)
    a_beta = np.arange(0, np.pi, step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)
    a_gamma, a_beta = a_gamma.flatten(), a_beta.flatten()

    F1 = np.array([objective_fun([a_gamma[i], a_beta[i]], Graph, approximation_List, p) for i in range(len(a_gamma))])

    # Grid search for the minimizing variables
    result = np.where(F1 == np.amin(F1))
    gamma, beta = a_gamma[result[0][0]], a_beta[result[0][0]]

    # Plot the expetation value F1
    if plot or fname:
        fig = plt.figure()
        #ax  = fig.gca(projection='3d')

        size = len(np.arange(0, np.pi, step_size))
        a_gamma, a_beta, F1 = a_gamma.reshape(size, size), a_beta.reshape(size, size), F1.reshape(size, size)
        #surf = ax.plot_surface(a_gamma, a_beta, F1, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        #ax.set_zlim(np.amin(F1)-1,np.amax(F1)+1)
        #ax.zaxis.set_major_locator(LinearLocator(5))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax = fig.add_subplot(1,1,1)
        ax.contourf(a_gamma, a_beta, F1, cmap=cm.coolwarm, antialiased=True)
    if (fname):
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

    return np.array([gamma, beta]), np.amin(F1)


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

def compareEpsilon(graph, epsilon_range):
    warm_means = []
    warm_means_energy = []
    warm_dev = []
    warm_energies = []

    RawBestCuts = bestGWcuts(graph, 3, 2, continuous=False, epsilon=0)  # get raw solutions using epsilon = 0
    print(RawBestCuts)
    p = 1

    epsilon_range = list(epsilon_range)
    for eps in epsilon_range:
        warmstart_cutsize = []
        warmstart_energy = []
        bestCuts = np.array([[epsilonFunction(cut[0], eps), cut[1]] for cut in deepcopy(RawBestCuts)], dtype=object)

        #bestCutsParams = [gridSearch(objectiveFunction, graph, bestCuts[i,0], p, step_size=0.2, show_plot=False)[0] for i in range(1)]
        optimizer_options = ({"rhobeg": 0.2, "disp": False})#, "maxiter": 10})# to limit optimizer iterations
        for i in range(len(bestCuts)):
            print(bestCuts[i])
            for j in range(3):
                params = [0, np.pi/2]
                params_warm_optimized = minimize(objectiveFunction, params, method="COBYLA", args=(graph, bestCuts[i,0], p), options=optimizer_options)
                energy, bestCut, maxCutChance = objectiveFunctionBest(params_warm_optimized.x, graph, bestCuts[i,0], p)
                warmstart_cutsize.append(bestCut)
                warmstart_energy.append(energy)
                print("params optimized: {} -> {}, energy measured: {}, cutsize: {}".format(params, params_warm_optimized.x, warmstart_energy[-1], warmstart_cutsize[-1]))
            print("{:.2f}%".format(100*(i+1+5*epsilon_range.index(eps))/(len(epsilon_range)*5)))

        warm_means.append(np.mean(warmstart_cutsize))
        warm_means_energy.append(np.mean(warmstart_energy))
        warm_dev.append([[eps for i in range(len(warmstart_cutsize))], warmstart_cutsize])
        warm_energies.append([[eps for i in range(len(warmstart_energy))], swapSign(warmstart_energy)])

    print(warmstart_cutsize)
    print(warm_means)
    warm_dev = np.array(warm_dev)
    warm_energies = np.array(warm_energies)
    plt.scatter(warm_dev[:,0], warm_dev[:,1], marker=".", color='gray', label="single cut")
    plt.scatter(warm_energies[:,0], warm_energies[:,1], marker=".", color='tan', label="single energy")
    plt.scatter(epsilon_range, warm_means, linestyle="None", marker="x", color="r", label="mean cut", alpha=.5)
    plt.scatter(epsilon_range, swapSign(warm_means_energy), linestyle="None", marker="x", color="darkorange", label="mean energy", alpha=.5)
    plt.legend(loc="best"), plt.xlabel("epsilon"), plt.ylabel("Energy/Cutsize"), plt.title("Warm-started QAOA comparison")
    plt.savefig("results/epsilons-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.close()


# graph = GraphGenerator.genButterflyGraph()
# graph = GraphGenerator.genGridGraph(4,4)
# graph = GraphGenerator.genFullyConnectedGraph(5)
# graph = GraphGenerator.genMustyGraph()
# graph = GraphGenerator.genRandomGraph(5,6)
graph = GraphGenerator.genWarmstartPaperGraph()
# GraphPlotter.plotGraph(nx.Graph(graph))

compareWarmStartEnergy(graph, [1,2,3 ], initialCut = [[0,0,1,1,1,1], 23], knownMaxCut = 27)
# compareOptimizerEnergy(graph, [1], ["Cobyla", "TNC"])  #TNC
# compareOptimizerEnergy(graph, [1,2], ["Cobyla", "Powell"])
# compareEpsilon(graph, np.arange(0.0, 0.51, 0.05))
# GraphPlotter.plotGraph(graph, fname="results/graph-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")

# print(params.x)
# plotCircuit(graph, params.x, p, FakeYorktown())
# plotSolution(graph, params.x, p)
# plt.plot(costs_history)
# plt.show()
