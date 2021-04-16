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
import os

def compareEpsilon(graph, rawBestCuts, epsilon_range, knownMaxCut=None):
    warm_means = []
    warm_means_energy = []
    warm_means_prob = []
    warm_dev = []
    warm_energies = []
    warm_probs = []
    rawResults = []

    p = 1

    epsilon_range = list(epsilon_range)
    for eps in epsilon_range:
        warmstart_cutsize = []
        warmstart_energy = []
        warmstart_prob = []
        bestCuts = np.array([[epsilonFunction(cut[0], eps), cut[1]] for cut in deepcopy(rawBestCuts)], dtype=object)

        #bestCutsParams = [gridSearch(objectiveFunction, graph, bestCuts[i,0], p, step_size=0.2, show_plot=False)[0] for i in range(1)]
        optimizer_options = ({"rhobeg": 0.2, "disp": False})#, "maxiter": 10})# to limit optimizer iterations
        for i in range(len(bestCuts)):
            print(bestCuts[i])
            for j in range(3):
                params = [0, np.pi/2]
                params_warm_optimized = minimize(objectiveFunction, params, method="COBYLA", args=(graph, bestCuts[i,0], p), options=optimizer_options)
                energy, bestCut, maxCutChance = objectiveFunctionBest(params_warm_optimized.x, graph, bestCuts[i,0], p, knownMaxCut=knownMaxCut)
                warmstart_cutsize.append(bestCut)
                warmstart_energy.append(energy)
                warmstart_prob.append(maxCutChance)
                print("params optimized: {} -> {}, energy measured: {}, cutsize: {}, max cut prob: {}".format(params, params_warm_optimized.x, warmstart_energy[-1], warmstart_cutsize[-1], maxCutChance))
                rawResults.append("{};{};{};{};{}".format(eps, bestCuts[i,0], energy, bestCut, maxCutChance))
            print("{:.2f}%".format(100*(i+1+5*epsilon_range.index(eps))/(len(epsilon_range)*5)))

        warm_means.append(np.median(warmstart_cutsize))
        warm_means_energy.append(np.median(warmstart_energy))
        warm_means_prob.append(np.mean(warmstart_prob))
        warm_dev.append([[eps for i in range(len(warmstart_cutsize))], warmstart_cutsize])
        warm_energies.append([[eps for i in range(len(warmstart_energy))], warmstart_energy])
        warm_probs.append([[eps for i in range(len(warmstart_energy))], warmstart_prob])

    print(warmstart_cutsize)
    print(warm_means)
    print(warm_energies)
    print(warm_means_energy)
    warm_dev = np.array(warm_dev)
    warm_energies = np.array(warm_energies)
    warm_probs = np.array(warm_probs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(warm_dev[:,0], warm_dev[:,1], marker=".", color='gray', label="single cut", alpha=.5)
    ax.scatter(epsilon_range, warm_means, linestyle="None", marker="x", color="r", label="median cut", alpha=.5)
    ax.scatter(warm_energies[:,0], warm_energies[:,1], marker=".", color='tan', label="single energy", alpha=.5)
    ax.scatter(epsilon_range, warm_means_energy, linestyle="None", marker="x", color="darkorange", label="median energy", alpha=.5)
    ax.set_xlabel("epsilon"), ax.set_ylabel("Energy/Cutsize"), ax.set_title("Warm-started QAOA comparison")

    ax2 = ax.twinx()
    ax2.scatter(warm_probs[:,0], warm_probs[:,1], marker=".", color="palegreen", label="single probability", alpha=.5)
    ax2.scatter(epsilon_range, warm_means_prob, marker="v", color="green", label="mean probability", alpha=.5)
    fig.subplots_adjust(bottom=0.22), fig.legend(loc="lower center", ncol=3), ax2.set_ylabel("max cut probability")

    add_to_name = "_"+str(len(bestCuts[0][0]))
    time =  datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + add_to_name
    path = os.getcwd() + "/results/" + time
    print ("The current working directory is %s" % path)
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    plt.savefig(path + "/epsilon"+add_to_name+".pdf", bbox_inches="tight")
    plt.savefig(path + "/epsilon"+add_to_name+".png", bbox_inches="tight")
    plt.close()

    rawResultsFile = open(path + "/raw"+add_to_name+".log", "w")
    rawResultsFile.write("\n".join(rawResults))
    rawResultsFile.close()

# graph = GraphGenerator.genFullyConnectedGraph(20)
# cuts = bestGWcuts(graph, 10, 5, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
# GraphStorage.store("graphs/fullyConnected-20-graph.txt", graph)
# GraphStorage.storeGWcuts("graphs/fullyConnected-20-cuts.txt", cuts)

graph_loaded = GraphStorage.load("graphs/fullyConnected-6-paperversion-graph.txt")
cuts_loaded = GraphStorage.loadGWcuts("graphs/fullyConnected-6-paperversion-cuts.txt")
maxcut = 27

print(graph_loaded.data)
print(cuts_loaded)

compareEpsilon(graph_loaded, cuts_loaded[:3], np.arange(0.0, 0.51, 0.25), knownMaxCut=maxcut)