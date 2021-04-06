from graphGenerator import GraphGenerator
from graphStorage import GraphStorage
from helperFunctions import epsilonFunction
from goemansWilliamson import bestGWcuts
from copy import deepcopy
from scipy.optimize import minimize
from maxcutQaoa import objectiveFunction, objectiveFunctionBest, totalCost, cost_function_C
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
        optimizer_options = ({"rhobeg": np.pi/2})  # ({"maxiter": 10})# to limit optimizer iterations

        for i in range(0, 1):

            # bestCutsParams = [gridSearch(objectiveFunction, graph, bestCuts[i,0], p)[0] for i in range(len(bestCuts))]
            for j in range(30):
                bestCut = bestCuts[i,0]
                if(initialCut):
                    bestCut = epsilonFunction(initialCut[0], epsilon=0.25)
                print(bestCut)
                params = np.zeros(2 * p)
                params = np.random.default_rng().uniform(0, np.pi, size=2*p)
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

        warm_MaxCutProb.append(np.median(warmstartMaxCutProb)*100)
        cold_MaxCutProb.append(np.median(coldstartMaxCutProb)*100)
        warm_MaxCutProb_Values.append([[p for i in range(len(warmstartMaxCutProb))], np.array(warmstartMaxCutProb)*100])
        cold_MaxCutProb_Values.append([[p for i in range(len(coldstartMaxCutProb))], np.array(coldstartMaxCutProb)*100])
        # TODO mean vs median
        warm_means.append(np.median(warmstart))
        cold_means.append(np.median(coldstart))
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
    usedCut = bestCuts[0,1]
    offset = totalCost(graph)
    if(initialCut):
        usedCut = initialCut[1]
    plt.plot([np.min(p_range), np.max(p_range)], [usedCut -offset, usedCut-offset], linestyle="dashed",
             label="used GW-Cut")
    plt.plot([np.min(p_range), np.max(p_range)], [knownMaxCut-offset, knownMaxCut-offset], linestyle="dashed",
             label="best GW-Cut")
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("Energy"), plt.title("Warm-started QAOA comparison")
    plt.savefig("results/warmstartEnergy-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.show()
    plt.close()

    #probabilitygraph
    warm_MaxCutProb_Values = np.array(warm_MaxCutProb_Values)
    cold_MaxCutProb_Values = np.array(cold_MaxCutProb_Values)
    plt.scatter(warm_MaxCutProb_Values[:,0], warm_MaxCutProb_Values[:,1], marker=".", color='red', label="warmstarted", alpha=.4)
    plt.scatter(cold_MaxCutProb_Values[:,0], cold_MaxCutProb_Values[:,1], marker=".", color='blue', label="coldstarted", alpha=.4)
    plt.scatter(p_range, warm_MaxCutProb, linestyle="None", marker="x", color="r", label="warm mean", alpha=.75)
    plt.scatter(p_range, cold_MaxCutProb, linestyle="None", marker="x", color="b", label="cold mean", alpha=.75)
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("MaxCut Probabilityin %"), plt.title(
        "MaxCut Probability")
    plt.savefig("results/warmstartEnergyProbability-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.show()
    plt.close()

def coldStartQAOA(graph, p_range, knownMaxCut = None):
    cold_means = []
    cold_value_list = []
    cold_max = []
    cold_MaxCutProb = []
    cold_MaxCutProb_Values = []

    bestCuts = bestGWcuts(graph, 8, 5, continuous=False, epsilon=0)
    bestCuts = np.array([[epsilonFunction(cut[0], epsilon=0.25), cut[1]] for cut in deepcopy(bestCuts)], dtype=object)
    if not knownMaxCut:
        knownMaxCut = bestCuts[len(bestCuts)-1,1]
    print("knownmaxcut {}".format(knownMaxCut))
    print(bestCuts)

    p_range = list(p_range)
    for p in p_range:
        coldstart = []
        coldstartMaxCutProb = []

        optimizer_options = None  # ({"maxiter": 10})# to limit optimizer iterations
        optimizer_options = ({"rhobeg": np.pi/2})  # ({"maxiter": 10})# to limit optimizer iterations

        for i in range(0, 1):

            # bestCutsParams = [gridSearch(objectiveFunction, graph, bestCuts[i,0], p)[0] for i in range(len(bestCuts))]
            for j in range(10):
                params = np.zeros(2 * p)
                params = np.random.default_rng().uniform(0, np.pi, size=2*p)
                               # plotCircuit(graph, bestCuts[i,0], params_warm_optimized.x, p,)
                params_cold_optimized = minimize(objectiveFunction, params, method="COBYLA", args=(graph, None, p),
                                                 options=optimizer_options)
                energyCold, cutCold, maxCutChanceCold = objectiveFunctionBest(params_cold_optimized.x, graph, None, p,
                                                                              knownMaxCut= knownMaxCut,
                                                                              showHistogram=True)
                print(energyCold)
                print("maxcutchancecold {}".format(maxCutChanceCold))
                coldstartMaxCutProb.append(maxCutChanceCold)
                coldstart.append(energyCold)
            print("{:.2f}%".format(100 * (i + 1 + 5 * p_range.index(p)) / (len(p_range) * 5)))


        cold_MaxCutProb.append(np.median(coldstartMaxCutProb)*100)
        cold_MaxCutProb_Values.append([[p for i in range(len(coldstartMaxCutProb))], np.array(coldstartMaxCutProb)*100])
        # TODO mean vs median
        cold_means.append(np.median(coldstart))
        cold_value_list.append([[p for i in range(len(coldstart))], coldstart])
        cold_max.append(np.min(coldstart))
        print(coldstart)

    #energygraph
    cold_value_list = np.array(cold_value_list)
    plt.scatter(cold_value_list[:,0], cold_value_list[:,1], marker=".", color='blue', label="coldstarted", alpha=.4)
    plt.scatter(p_range, cold_means, linestyle="None", marker="x", color="b", label="mean cut", alpha=.75)
    offset = totalCost(graph)
    plt.plot([np.min(p_range), np.max(p_range)], [knownMaxCut-offset, knownMaxCut-offset], linestyle="dashed",
             label="best GW-Cut")
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("Energy"), plt.title("Warm-started QAOA comparison")
    plt.savefig("results/coldstartEnergy-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.show()
    plt.close()

    #probabilitygraph
    cold_MaxCutProb_Values = np.array(cold_MaxCutProb_Values)
    plt.scatter(cold_MaxCutProb_Values[:,0], cold_MaxCutProb_Values[:,1], marker=".", color='blue', label="coldstarted", alpha=.4)
    plt.scatter(p_range, cold_MaxCutProb, linestyle="None", marker="x", color="b", label="cold mean", alpha=.75)
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("MaxCut Probabilityin %"), plt.title(
        "MaxCut Probability")
    plt.savefig("results/coldstartProbability-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.show()
    plt.close()


def testOptimizerDevelopmentExperiment():
    p =3
    params = np.random.default_rng().uniform(0, np.pi, size=2)
    setparams = [-np.pi,-np.pi]
    energy = []
    for i in range(20):
        optimizer_options = {"rhobeg": np.pi/2}
        params = np.random.uniform(0, 2*np.pi, size=2*(p-1))
        # params = [0,0] * (p-1)
        optimizedparams = minimize(objectiveFunction, params, method="COBYLA",
                                   args=(graph_loaded, cuts_loaded[2,0], p, setparams), options=optimizer_options)
        optimizedparams = [setparams[0], setparams[1]] + list(optimizedparams.x)

        energy.append( objectiveFunctionBest(optimizedparams, graph_loaded, cuts_loaded[2,0], p, 27)[0])
        # x = minimize(objectiveFunction, params, method="COBYLA",
        #                              args=(graph_loaded, cuts_loaded[2,0], 3))
        # results.append(objectiveFunctionBest(x.x, graph_loaded, cuts_loaded[2,0], 3,
        #                                      knownMaxCut= 27,
        #                                      showHistogram=False))
        print(optimizedparams)
    print(energy)

# graph = GraphGenerator.genButterflyGraph()
# graph = GraphGenerator.genGridGraph(4,4)
# graph = GraphGenerator.genFullyConnectedGraph(17)
# graph = GraphGenerator.genMustyGraph()
# graph = GraphGenerator.genRandomGraph(5,6)
# graph = GraphGenerator.genWarmstartPaperGraph()
# GraphPlotter.plotGraph(nx.Graph(graph))

graph_loaded = GraphStorage.load("graphs/fullyConnected-6-paperversion-graph.txt")
cuts_loaded = GraphStorage.loadGWcuts("graphs/fullyConnected-6-paperversion-cuts.txt")
epsilon = 0.25
cuts_loaded = np.array([[epsilonFunction(cut[0], epsilon), cut[1]] for cut in cuts_loaded], dtype=object)

# graph_loaded = GraphGenerator.genDiamondGraph()

print(cuts_loaded)

# compareWarmStartEnergy(graph_loaded, [1,2,3,4,5 ], initialCut = [[0,1,0,1], 4], knownMaxCut = 4)
# compareWarmStartEnergy(graph_loaded, [1,2,3 ], initialCut = [[0,0,1,1,1,1], 23], knownMaxCut = 27)
# coldStartQAOA(graph_loaded, [1,2,3], knownMaxCut=4)


# compareWarmStartEnergy(graph_loaded, [1,2,3 ])

# GraphPlotter.plotGraph(graph, fname="results/graph-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")

# print(params.x)
# plotCircuit(graph, params.x, p, FakeYorktown())
# plotSolution(graph, params.x, p)
# plt.plot(costs_history)
# plt.show()
