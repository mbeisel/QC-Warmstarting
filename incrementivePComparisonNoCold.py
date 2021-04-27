from matplotlib import cm

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
    warm_value_list = []
    warm_max = []
    warm_MaxCutProb = []
    warm_MaxCutProb_Values = []

    print("knownmaxcut {}".format(knownMaxCut))


    p_range = list(p_range)
    bestParamsForP = [[0,0] for i in range(len(p_range))]
    for count,p in enumerate(p_range):
        warmstart = []
        warmstartMaxCutProb = []
        optimizer_options = None  # ({"maxiter": 10})# to limit optimizer iterations
        # optimizer_options = ({"rhobeg": np.pi/2})  # ({"maxiter": 10})# to limit optimizer iterations

        for i in range(0, 1):
            #optimize j times starting with different startvalues
            for j in range(2):
                bestCut = epsilonFunction(initialCut[0], epsilon=epsilon)
                print(bestCut)
                params = np.random.default_rng().uniform(0, np.pi, size=2*p)
                params = np.zeros(2*p)

                if(bestParamsForP[count-1][0] != 0):
                    for e in range(p_range[count-1]*2):
                          params[e] = bestParamsForP[count-1][1][e]

                energyWarmList, cutWarmList, maxCutChanceWarmList= [], [], []
                #optimize k times with the same startvalues and take the best
                for k in range(1):
                    params_warm_optimized = minimize(objectiveFunction, params[p_range[count-1]*2:] if p > 1 else params, method="Cobyla",
                                                     args=(graph, bestCut, p, list(params[:p_range[count-1]*2]) if p > 1 else None, initialCut[1]), options=optimizer_options)
                    if p > 1:
                        params_warm_optimized.x = list(params[:p_range[count-1]*2]) + list(params_warm_optimized.x)

                    energyWarm, cutWarm, maxCutChanceWarm = objectiveFunctionBest(params_warm_optimized.x, graph, bestCut, p,
                                                                                  knownMaxCut= knownMaxCut,
                                                                                  showHistogram=False, inputCut=initialCut[1])

                    if bestParamsForP[count][0] < energyWarm:
                        bestParamsForP[count][0] = energyWarm
                        bestParamsForP[count][1] = list(params_warm_optimized.x)
                    energyWarmList.append(energyWarm)
                    cutWarmList.append(cutWarm)
                    maxCutChanceWarmList.append(maxCutChanceWarm)
                print(energyWarmList)
                warmstart.append(np.max(energyWarmList))
                warmstartMaxCutProb.append(np.max(maxCutChanceWarmList))
                print("maxcutchance {}".format(np.max(maxCutChanceWarmList)))

            print("{:.2f}%".format(100 * (i + 1 + 5 * p_range.index(p)) / (len(p_range) * 5)))

        warm_MaxCutProb.append(np.median(warmstartMaxCutProb)*100)
        warm_MaxCutProb_Values.append([[p for i in range(len(warmstartMaxCutProb))], np.array(warmstartMaxCutProb)*100])
        warm_means.append(np.median(warmstart))
        warm_value_list.append([[p for i in range(len(warmstart))], warmstart])
        warm_max.append(np.min(warmstart))
        print(warmstart)
        print(bestParamsForP)

    print([warm_means])
    print([warm_MaxCutProb])
    # print([warm_max, cold_max])


    #energygraph
    warm_value_list = np.array(warm_value_list)

    plt.scatter(warm_value_list[:,0], warm_value_list[:,1], marker=".", color='red', label="warmstarted", alpha=.4)
    plt.scatter(p_range, warm_means, linestyle="None", marker="x", color="r", label="median cut", alpha=.75)
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
    plt.scatter(warm_MaxCutProb_Values[:,0], warm_MaxCutProb_Values[:,1], marker=".", color='red', label="warmstarted", alpha=.4)
    plt.scatter(p_range, warm_MaxCutProb, linestyle="None", marker="x", color="r", label="warm median", alpha=.75)
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("MaxCut Probabilityin %"), plt.title(
        "MaxCut Probability")
    plt.savefig("results/warmstartEnergyProbability-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    plt.show()
    plt.close()


def compareWarmStartEnergyMethods(graph, p_range, initialCut, knownMaxCut = None, onlyOptimizeCurrentP = False, epsilon =0.25):
    n_methods = 5
    warm_means = []
    warm_value_list = []
    warm_max = []
    warm_MaxCutProb = [[] for i in range(n_methods)]
    warm_MaxCutProb_Values = []

    p_range = list(p_range)
    bestParamsForP = [[[0,0] for i in range(len(p_range))] for i in range(n_methods)]
    for count,p in enumerate(p_range):
        warmstart = [[] for i in range(n_methods)]
        warmstartMaxCutProb = [[] for i in range(n_methods)]
        optimizer_options = None  # ({"maxiter": 10})# to limit optimizer iterations
        # optimizer_options = ({"rhobeg": np.pi/2})  # ({"maxiter": 10})# to limit optimizer iterations

        for i in range(0, 1):
            #optimize j times starting with different startvalues
            for j in range(5):
                bestCut = epsilonFunction(initialCut[0], epsilon=epsilon)

                energyWarmList, cutWarmList, maxCutChanceWarmList= [[] for i in range(n_methods)], [[] for i in range(n_methods)], [[] for i in range(n_methods)]
                for methodCount in range(n_methods):
                    params = np.zeros(2*p)
                    params = np.random.default_rng().uniform(0, np.pi, size=2*p)

                    if(p > 1):
                        if(bestParamsForP[methodCount][count-1][0] != 0):
                            for e in range(p_range[count-1]*2):
                                params[e] = bestParamsForP[methodCount][count-1][1][e]


                    #optimize k times with the same startvalues and take the best
                    for k in range(1):
                        params_warm_optimized = minimize(objectiveFunction, params[p_range[count-1]*2:] if p > 1 else params, method="Cobyla",
                                                         args=(graph, bestCut, p, list(params[:p_range[count-1]*2]) if p > 1 else None, initialCut[1], methodCount), options=optimizer_options)
                        if p > 1:
                            params_warm_optimized.x = list(params[:p_range[count-1]*2]) + list(params_warm_optimized.x)

                        energyWarm, cutWarm, maxCutChanceWarm = objectiveFunctionBest(params_warm_optimized.x, graph, bestCut, p,
                                                                                      knownMaxCut= knownMaxCut,
                                                                                      showHistogram=False, inputCut=initialCut[1], method=methodCount)

                        if bestParamsForP[methodCount][count][0] < energyWarm:
                            bestParamsForP[methodCount][count][0] = energyWarm
                            bestParamsForP[methodCount][count][1] = list(params_warm_optimized.x)
                        energyWarmList[methodCount].append(energyWarm)
                        cutWarmList[methodCount].append(cutWarm)
                        maxCutChanceWarmList[methodCount].append(maxCutChanceWarm)
                    print(energyWarmList)
                    warmstart[methodCount].append(np.max(energyWarmList[methodCount]))
                    warmstartMaxCutProb[methodCount].append(np.max(maxCutChanceWarmList[methodCount]))
                    print("maxcutchance {} at j={}".format(np.max(maxCutChanceWarmList[methodCount]), j))

            print("{:.2f}%".format(100 * (i + 1 + 5 * p_range.index(p)) / (len(p_range) * 5)))

        print("WARMSTARTPROB")
        print(warmstartMaxCutProb)
        for h in range(n_methods):
            warm_MaxCutProb[h].append(np.median(warmstartMaxCutProb[h])*100)
        warm_MaxCutProb_Values.append([[p for i in range(len(warmstartMaxCutProb))], np.array(warmstartMaxCutProb)*100])
        warm_means.append(np.median(warmstart))
        warm_value_list.append([[p for i in range(len(warmstart))], warmstart])
        warm_max.append(np.min(warmstart))
        print(warmstart)
        print(bestParamsForP)

    print([warm_means])
    print([warm_MaxCutProb])
    # print([warm_max, cold_max])


    #energygraph
    warm_value_list = np.array(warm_value_list)


    # plt.scatter(warm_value_list[:,0], warm_value_list[:,1], marker=".", color='red', label="warmstarted", alpha=.4)
    # plt.scatter(p_range, warm_means, linestyle="None", marker="x", color="r", label="median cut", alpha=.75)
    # offset = totalCost(graph)
    # if(initialCut):
    #     usedCut = initialCut[1]
    # plt.plot([np.min(p_range), np.max(p_range)], [usedCut -offset, usedCut-offset], linestyle="dashed",
    #          label="used GW-Cut")
    # plt.plot([np.min(p_range), np.max(p_range)], [knownMaxCut-offset, knownMaxCut-offset], linestyle="dashed",
    #          label="best GW-Cut")
    # plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("Energy"), plt.title("Warm-started QAOA comparison")
    # plt.savefig("results/warmstartEnergy-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png", format="png")
    # plt.show()
    # plt.close()

    methodValues = [[] for methodCount in range(n_methods)]
    #probabilitygraph
    print(warm_MaxCutProb_Values)
    for p in range(len(p_range)):
        for methodCount in range(n_methods):
            # [methodValues[methodCount][0].append(p) for i in range(len(warm_MaxCutProb_Values[p][1][methodCount]))]
            # [methodValues[methodCount][1].append(e) for e in warm_MaxCutProb_Values[p][1][methodCount]]
            [methodValues[methodCount].append([p_range[p],e]) for i,e in enumerate(warm_MaxCutProb_Values[p][1][methodCount])]


    methodValues = np.array(methodValues)
    colors = cm.get_cmap("rainbow", n_methods)


    for methodCount in range(n_methods):
        plt.scatter(methodValues[methodCount][:,0], methodValues[methodCount][:,1], marker=".", color = colors(methodCount), label="Method" +str(methodCount), alpha=.4)
        plt.scatter(p_range, warm_MaxCutProb[methodCount], linestyle="None", marker="x",color = colors(methodCount), alpha=.8)
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("MaxCut Probabilityin %"), plt.title(
        "MaxCut Probability")
    plt.savefig("results/compareProbabilityMethods-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+"_{}_{}.png".format(graph.shape[0], initialCut[1]), format="png")
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

# graph_loaded = GraphStorage.load("graphs/fullyConnected-12-graph.txt")
# cuts_loaded = GraphStorage.loadGWcuts("graphs/fullyConnected-12-cuts.txt")


# graph_loaded = GraphGenerator.genDiamondGraph()

print(cuts_loaded)

# compareWarmStartEnergy(graph_loaded, [1,2,3,4,5 ], initialCut = [[0,1,0,1], 4], knownMaxCut = 4)
# compareWarmStartEnergy(graph_loaded, [1,2,3], initialCut = [[0,0,1,1,1,1], 23], knownMaxCut = 27, epsilon=0.325)
compareWarmStartEnergyMethods(graph_loaded, [1,2,3,4], initialCut = [[0,0,1,1,1,1], 23], knownMaxCut = 27, epsilon=0.325)
# compareWarmStartEnergy(graph_loaded, [1,2], initialCut = cuts_loaded[0],  knownMaxCut = 95, epsilon=0.325)
# compareWarmStartEnergyMethods(graph_loaded, [1], initialCut = [[0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0], 88.0],  knownMaxCut = 95, epsilon=0.325)
# compareWarmStartEnergyMethods(graph_loaded, [1], initialCut = cuts_loaded[1],  knownMaxCut = 95, epsilon=0.325)
# coldStartQAOA(graph_loaded, [1,2,3], knownMaxCut=4)
# compareWarmStartEnergyMethods(graph_loaded, [1], initialCut = [[1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], 78], knownMaxCut = 95, epsilon=0.5)



# compareWarmStartEnergy(graph_loaded, [1,2,3 ])

# GraphPlotter.plotGraph(graph, fname="results/graph-"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")

# print(params.x)
# plotCircuit(graph, params.x, p, FakeYorktown())
# plotSolution(graph, params.x, p)
# plt.plot(costs_history)
# plt.show()
