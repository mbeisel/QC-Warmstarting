from helperFunctions import epsilonFunction
from goemansWilliamson import bestGWcuts
from copy import deepcopy
from scipy.optimize import minimize
from maxcutQaoa import objectiveFunction, objectiveFunctionBest, cost_function_C
import numpy as np
import matplotlib.pyplot as plt
from graphGenerator import GraphGenerator
from graphStorage import GraphStorage
from datetime import datetime

def compareEpsilon(graph, rawBestCuts, epsilon_range):
    warm_means = []
    warm_means_energy = []
    warm_dev = []
    warm_energies = []

    p = 1

    epsilon_range = list(epsilon_range)
    for eps in epsilon_range:
        warmstart_cutsize = []
        warmstart_energy = []
        bestCuts = np.array([[epsilonFunction(cut[0], eps), cut[1]] for cut in deepcopy(rawBestCuts)], dtype=object)

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
        warm_energies.append([[eps for i in range(len(warmstart_energy))], warmstart_energy])

    print(warmstart_cutsize)
    print(warm_means)
    print(warm_energies)
    print(warm_means_energy)
    warm_dev = np.array(warm_dev)
    warm_energies = np.array(warm_energies)
    plt.scatter(warm_dev[:,0], warm_dev[:,1], marker=".", color='gray', label="single cut")
    plt.scatter(warm_energies[:,0], warm_energies[:,1], marker=".", color='tan', label="single energy")
    plt.scatter(epsilon_range, warm_means, linestyle="None", marker="x", color="r", label="mean cut", alpha=.5)
    plt.scatter(epsilon_range, warm_means_energy, linestyle="None", marker="x", color="darkorange", label="mean energy", alpha=.5)
    plt.legend(loc="best"), plt.xlabel("epsilon"), plt.ylabel("Energy/Cutsize"), plt.title("Warm-started QAOA comparison")
    plt.savefig("results/epsilons-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.close()

# graph = GraphGenerator.genFullyConnectedGraph(20)
# cuts = bestGWcuts(graph, 10, 5, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
# GraphStorage.store("graphs/fullyConnected-20-graph.txt", graph)
# GraphStorage.storeGWcuts("graphs/fullyConnected-20-cuts.txt", cuts)

graph_loaded = GraphStorage.load("graphs/fullyConnected-20-graph.txt")
cuts_loaded = GraphStorage.loadGWcuts("graphs/fullyConnected-20-cuts.txt")

print(graph_loaded.data)
print(cuts_loaded)
compareEpsilon(graph_loaded, cuts_loaded, np.arange(0.0, 0.50, 0.25))