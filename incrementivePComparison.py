from graphStorage import GraphStorage
from helperFunctions import epsilonFunction
from goemansWilliamson import bestGWcuts
from copy import deepcopy
from scipy.optimize import minimize
from maxcutQaoa import objectiveFunction, objectiveFunctionBest, totalCost, cost_function_C
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def compareWarmStartEnergy(graph, p_range, initialCut, knownMaxCut = None, onlyOptimizeCurrentP = False, epsilon =0.25):
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

    p_range = list(p_range)
    bestParamsForPwarm = [[-999999999, None] for i in range(len(p_range))]
    bestParamsForPcold = [[-999999999,None] for i in range(len(p_range))]
    for count,p in enumerate(p_range):
        warmstart = []
        coldstart = []
        warmstartMaxCutProb = []
        coldstartMaxCutProb = []

        optimizer_options = None  # ({"maxiter": 10})# to limit optimizer iterations
        # optimizer_options = ({"rhobeg": np.pi/2})  # ({"maxiter": 10})# to limit optimizer iterations

        for i in range(0, 1):


            #optimize j times starting with different startvalues or iteratively with best from previous run
            for j in range(2):
                bestCut = epsilonFunction(initialCut[0], epsilon=epsilon)

                params = np.random.default_rng().uniform(0, np.pi, size=2*p)
                params = np.zeros(2*p)
                params_cold = np.zeros(2*p)
                if(bestParamsForPwarm[count-1][0] != -999999999):
                    if p > 1:
                        for e in range(p_range[count-1]*2):
                              params[e] = bestParamsForPwarm[count-1][1][e]

                if(bestParamsForPcold[count-1][0] != -999999999):
                    # params_cold[e] = bestParamsForPcold[count-1][]
                    if p > 1:
                        for e in range(p_range[count-1]*2):
                            params_cold[e] = bestParamsForPcold[count-1][1][e]

                energyWarmList, cutWarmList, maxCutChanceWarmList, energyColdList, cutColdList, maxCutChanceColdList = [], [], [], [], [], []
                #optimize k times with the same startvalues and take the best
                for k in range(1):

                    params_warm_optimized = minimize(objectiveFunction, params[p_range[count-1]*2:] if p > 1 else params, method="COBYLA",
                                                     args=(graph, bestCut, p, list(params[:p_range[count-1]*2]) if p > 1 else None, initialCut[1]), options=optimizer_options)
                    params_cold_optimized = minimize(objectiveFunction, params_cold, method="COBYLA", args=(graph, None, p),
                                                     options=optimizer_options)
                    if p > 1:
                        params_warm_optimized.x = list(params[:p_range[count-1]*2]) + list(params_warm_optimized.x)

                    energyWarm, cutWarm, maxCutChanceWarm = objectiveFunctionBest(params_warm_optimized.x, graph, bestCut, p,
                                                                                  knownMaxCut= knownMaxCut,
                                                                                  showHistogram=False, inputCut=initialCut[1])
                    energyCold, cutCold, maxCutChanceCold = objectiveFunctionBest(params_cold_optimized.x, graph, None, p,
                                                                                  knownMaxCut= knownMaxCut,
                                                                                  showHistogram=False)
                    if bestParamsForPwarm[count][0] < energyWarm:
                        bestParamsForPwarm[count][0] = energyWarm
                        bestParamsForPwarm[count][1] = list(params_warm_optimized.x)
                    if bestParamsForPcold[count][0] < energyCold:
                        bestParamsForPcold[count][0] = energyCold
                        bestParamsForPcold[count][1] = list(params_cold_optimized.x)
                    energyWarmList.append(energyWarm)
                    cutWarmList.append(cutWarm)
                    maxCutChanceWarmList.append(maxCutChanceWarm)
                    energyColdList.append(energyCold)
                    cutColdList.append(cutCold)
                    maxCutChanceColdList.append(maxCutChanceCold)
                print(energyWarmList)
                warmstart.append(np.max(energyWarmList))
                warmstartMaxCutProb.append(np.max(maxCutChanceWarmList))
                print("maxcutchance {}".format(np.max(maxCutChanceWarmList)))

                print("maxcutchancecold {}".format(np.max(maxCutChanceColdList)))
                coldstartMaxCutProb.append(np.max(maxCutChanceColdList))
                coldstart.append(np.max(energyColdList))
            print("{:.2f}%".format(100 * (i + 1 + 5 * p_range.index(p)) / (len(p_range) * 5)))

        warm_MaxCutProb.append(np.median(warmstartMaxCutProb)*100)
        cold_MaxCutProb.append(np.median(coldstartMaxCutProb)*100)
        warm_MaxCutProb_Values.append([[p for i in range(len(warmstartMaxCutProb))], np.array(warmstartMaxCutProb)*100])
        cold_MaxCutProb_Values.append([[p for i in range(len(coldstartMaxCutProb))], np.array(coldstartMaxCutProb)*100])
        warm_means.append(np.median(warmstart))
        cold_means.append(np.median(coldstart))
        warm_value_list.append([[p for i in range(len(warmstart))], warmstart])
        cold_value_list.append([[p for i in range(len(coldstart))], coldstart])
        warm_max.append(np.min(warmstart))
        cold_max.append(np.min(coldstart))
        print(warmstart)
        print(coldstart)
        print(bestParamsForPwarm)
        print(bestParamsForPcold)

    print([warm_means, cold_means])
    print([warm_MaxCutProb, cold_MaxCutProb])
    # print([warm_max, cold_max])


    #energygraph
    warm_value_list = np.array(warm_value_list)
    cold_value_list = np.array(cold_value_list)
    plt.scatter(warm_value_list[:,0], warm_value_list[:,1], marker=".", color='red', label="warmstarted", alpha=.4)
    plt.scatter(cold_value_list[:,0], cold_value_list[:,1], marker=".", color='blue', label="coldstarted", alpha=.4)
    plt.scatter(p_range, warm_means, linestyle="None", marker="x", color="r", label="median cut", alpha=.75)
    plt.scatter(p_range, cold_means, linestyle="None", marker="x", color="b", label="median cut", alpha=.75)
    offset = totalCost(graph)
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
    plt.scatter(p_range, warm_MaxCutProb, linestyle="None", marker="x", color="r", label="warm median", alpha=.75)
    plt.scatter(p_range, cold_MaxCutProb, linestyle="None", marker="x", color="b", label="cold median", alpha=.75)
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("MaxCut Probabilityin %"), plt.title(
        "MaxCut Probability")
    plt.savefig("results/warmstartEnergyProbability-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.show()
    plt.close()


# graph = GraphGenerator.genButterflyGraph()
# graph = GraphGenerator.genGridGraph(4,4)
# graph = GraphGenerator.genFullyConnectedGraph(17)
# graph = GraphGenerator.genMustyGraph()
# graph = GraphGenerator.genRandomGraph(5,6)
# graph = GraphGenerator.genWarmstartPaperGraph()
# GraphPlotter.plotGraph(nx.Graph(graph))

graph_loaded = GraphStorage.load("graphs/fullyConnected-6-paperversion-graph.txt")
cuts_loaded = GraphStorage.loadGWcuts("graphs/fullyConnected-6-paperversion-cuts.txt")
#
# graph_loaded = GraphStorage.load("graphs/fullyConnected-12-graph.txt")
# cuts_loaded = GraphStorage.loadGWcuts("graphs/fullyConnected-12-cuts.txt")


# graph_loaded = GraphGenerator.genDiamondGraph()

print(cuts_loaded)

# compareWarmStartEnergy(graph_loaded, [1,2,3,4,5 ], initialCut = [[0,1,0,1], 4], knownMaxCut = 4)
compareWarmStartEnergy(graph_loaded, [1,2,3,4,5], initialCut = [[0,0,1,1,1,1], 23], knownMaxCut = 27, epsilon=0.325)
# compareWarmStartEnergy(graph_loaded, [1,2], initialCut = cuts_loaded[0],  knownMaxCut = 95, epsilon=0.325)
# coldStartQAOA(graph_loaded, [1,2,3], knownMaxCut=4)


# compareWarmStartEnergy(graph_loaded, [1,2,3 ])

# GraphPlotter.plotGraph(graph, fname="results/graph-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")

# print(params.x)
# plotCircuit(graph, params.x, p, FakeYorktown())
# plotSolution(graph, params.x, p)
# plt.plot(costs_history)
# plt.show()
