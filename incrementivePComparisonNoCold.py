import os

from matplotlib import cm

from graphGenerator import GraphGenerator
from graphStorage import GraphStorage
from helperFunctions import epsilonFunction
from goemansWilliamson import bestGWcuts
from copy import deepcopy, copy
from scipy.optimize import minimize
from maxcutQaoa import objectiveFunction, objectiveFunctionBest, totalCost, cost_function_C
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from MinimizeWrapper import MinimizeWrapper

def compareWarmStartEnergy(graph, p_range, initialCut, knownMaxCut = None, onlyOptimizeCurrentP = False, epsilon =0.25, energymethod = 0):
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

                if(p > 1):
                    if(bestParamsForP[count-1][0] != 0):
                        for e in range(p_range[count-1]*2):
                            params[e] = bestParamsForP[count-1][1][e]

                energyWarmList, cutWarmList, maxCutChanceWarmList= [], [], []
                #optimize k times with the same startvalues and take the best
                for k in range(1):
                    params_warm_optimized = MinimizeWrapper().minimize(objectiveFunction, params[p_range[count-1]*2:] if p > 1 else params, method="Cobyla",
                                                     args=(graph, bestCut, p, list(params[:p_range[count-1]*2]) if p > 1 else None, initialCut[1], energymethod), options=optimizer_options)
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


def compareWarmStartEnergyMethods(iterations, graph, p_range, initialCut, knownMaxCut = None, onlyOptimizeCurrentP = False, epsilon =0.25, doCold = False, methods = None, doIncremental=True, method_params = None, labels = None, useBestParmas = False):
    if(initialCut):
        print("Use: {}".format(initialCut))
    warm_means = []
    warm_value_list = []
    warm_MaxCutProb = [[] for i in range(len(methods))]
    warm_MaxCutProb_Values = []
    warm_BetterCutProb = [[] for i in range(len(methods))]
    warm_BetterCutProb_Values = []
    cold_means = []
    cold_MaxCutProb = []
    cold_MaxCutProb_Values = []
    cold_BetterCutProb = []
    cold_BetterCutProb_Values = []
    bestParamsForPcold = [[-999999999,None] for i in range(len(p_range))]
    optimizer_options = None
    raw_median_results = ["doincremental:{}; onlyOptimizeCurrentP:{}; useBestParams:{}".format(doIncremental, onlyOptimizeCurrentP, useBestParams)]
    raw_all_results =  ["doincremental:{}; onlyOptimizeCurrentP:{}; useBestParams:{}".format(doIncremental, onlyOptimizeCurrentP, useBestParams)]
    warmAllMethodParams = [[[] for i in range(iterations)] for i in range(len(methods))]
    coldAllParams = [[] for i in range(iterations)]


    p_range = list(p_range)
    bestParamsForP = [[[-999999999,None] for i in range(len(p_range))] for i in range(len(methods))]
    for count,p in enumerate(p_range):
        warmstart = [[] for i in range(len(methods))]
        warmstartMaxCutProb = [[] for i in range(len(methods))]
        warmstartBetterCutProb = [[] for i in range(len(methods))]

        coldstart = []
        coldstartMaxCutProb = []
        coldstartBetterCutProb = []



        for i in range(0, 1):
            #optimize j times starting with different startvalues
            for j in range(iterations):
                params_raw = np.concatenate((np.random.default_rng().uniform(0, np.pi, size=2),np.zeros(2*(p-1))))
                if doIncremental and p > 1:
                    params_raw = np.zeros(2*p)
                if doCold ==True:
                    params_cold = copy(params_raw)

                    if( p > 1 and doIncremental and bestParamsForPcold[count-1][0] != -999999999):
                        if not useBestParmas:
                            for e in range(p_range[count-1]*2):
                                params_cold[e] = coldAllParams[j][e]
                        else:
                            for e in range(p_range[count-1]*2):
                                params_cold[e] = bestParamsForPcold[count-1][1][e]

                    if onlyOptimizeCurrentP == True:

                        params_cold_optimized = MinimizeWrapper().minimize(objectiveFunction, params_cold[p_range[count-1]*2:] if p > 1 else params_cold, method="Cobyla",
                                                         args=(graph, None, p, list(params_cold[:p_range[count-1]*2]) if p > 1 else None), options=optimizer_options)
                        if p > 1:
                            params_cold_optimized.bestValue[0] = list(params_cold[:p_range[count-1]*2]) + list(params_cold_optimized.bestValue[0])
                    else:
                        params_cold_optimized = MinimizeWrapper().minimize(objectiveFunction, params_cold, method="COBYLA", args=(graph, None, p),
                                                                 options=optimizer_options)
                    energyCold, cutCold, maxCutChanceCold, betterCutChanceCold = objectiveFunctionBest(params_cold_optimized.bestValue[0], graph, None, p,
                                                                                  knownMaxCut= knownMaxCut,
                                                                                  showHistogram=False)
                    if bestParamsForPcold[count][0] < energyCold:
                        bestParamsForPcold[count][0] = energyCold
                        bestParamsForPcold[count][1] = list(params_cold_optimized.bestValue[0])
                    if not useBestParmas:
                        coldAllParams[j] = list(params_cold_optimized.bestValue[0])
                    coldstart.append(energyCold)
                    coldstartMaxCutProb.append(maxCutChanceCold)
                    coldstartBetterCutProb.append(betterCutChanceCold)
                    print("maxcutchance for coldstart:{} at j={}".format(maxCutChanceCold, j))
                    raw_all_results.append("{};{},{};{};{};{};{}".format(p,j, "cold", energyCold, maxCutChanceCold, 0, ','.join(list(str(e) for e in params_cold_optimized.bestValue[0]))))





                bestCut = epsilonFunction(initialCut[0], epsilon=epsilon)
                energyWarmList, cutWarmList, maxCutChanceWarmList, betterCutChanceWarmList, paramsWarmList= [[] for i in range(len(methods))], [[] for i in range(len(methods))], [[] for i in range(len(methods))], [[] for i in range(len(methods))],[[] for i in range(len(methods))]

                for methodCount, method in enumerate(methods):
                    params = copy(params_raw)

                    if(p > 1 and doIncremental and bestParamsForP[methodCount][count-1][0] != -999999999):
                        if not useBestParmas:
                            for e in range(p_range[count-1]*2):
                                params[e] = warmAllMethodParams[methodCount][j][e]
                        else:
                            for e in range(p_range[count-1]*2):
                                params[e] = bestParamsForP[methodCount][count-1][1][e]

                    #optimize k times with the same startvalues and take the best
                    for k in range(1):
                        if onlyOptimizeCurrentP == True:

                            params_warm_optimized = MinimizeWrapper().minimize(objectiveFunction, params[p_range[count-1]*2:] if p > 1 else params, method="Cobyla",
                                                             args=(graph, bestCut, p, list(params[:p_range[count-1]*2]) if p > 1 else None, initialCut[1], method, method_params[methodCount]), options=optimizer_options)
                            if p > 1:
                                params_warm_optimized.bestValue[0] = list(params[:p_range[count-1]*2]) + list(params_warm_optimized.bestValue[0])
                        else:
                            params_warm_optimized = MinimizeWrapper().minimize(objectiveFunction, params, method="Cobyla",
                                                             args=(graph, bestCut, p, None, initialCut[1], method, method_params[methodCount]), options=optimizer_options)

                        energyWarm, cutWarm, maxCutChanceWarm, betterCutChanceWarm = objectiveFunctionBest(params_warm_optimized.bestValue[0], graph, bestCut, p,
                                                                                      knownMaxCut= knownMaxCut,
                                                                                      showHistogram=False, inputCut=initialCut[1], method=method, method_params=method_params[methodCount])


                        if bestParamsForP[methodCount][count][0] < energyWarm:
                            bestParamsForP[methodCount][count][0] = energyWarm
                            bestParamsForP[methodCount][count][1] = list(params_warm_optimized.bestValue[0])
                        if not useBestParmas:
                            warmAllMethodParams[methodCount][j] = list(params_warm_optimized.bestValue[0])
                        energyWarmList[methodCount].append(energyWarm)
                        cutWarmList[methodCount].append(cutWarm)
                        maxCutChanceWarmList[methodCount].append(maxCutChanceWarm)
                        betterCutChanceWarmList[methodCount].append(betterCutChanceWarm)
                        paramsWarmList[methodCount].append(list(params_warm_optimized.bestValue[0]))
                    warmstart[methodCount].append(np.max(energyWarmList[methodCount]))
                    warmstartMaxCutProb[methodCount].append(np.max(maxCutChanceWarmList[methodCount]))
                    warmstartBetterCutProb[methodCount].append(np.max(betterCutChanceWarmList[methodCount]))
                    print("maxcutchance for method {}:{} at j={}".format(method, np.max(maxCutChanceWarmList[methodCount]), j))
                    raw_all_results.append("{};{};{};{};{};{};{}".format(p,j, method, np.max(energyWarmList[methodCount]), np.max(maxCutChanceWarmList[methodCount]), np.max(betterCutChanceWarmList[methodCount]), ','.join(list(str(e) for e in paramsWarmList[methodCount][0]))))


            print("{:.2f}%".format(100 * (i + 1 + 5 * p_range.index(p)) / (len(p_range) * 5)))

        print("WARMSTARTPROB")
        print(warmstartMaxCutProb)
        for h, method in enumerate(methods):
            warm_MaxCutProb[h].append(np.median(warmstartMaxCutProb[h])*100)
            warm_BetterCutProb[h].append(np.median(warmstartBetterCutProb[h])*100)
            # save p, method, energy_median, maxcutchance_median, bettercutchance_median
            raw_median_results.append("{};{};{};{};{}".format(p, methods[h], np.median(warmstart[h]), np.median(warmstartMaxCutProb[h]), np.median(warmstartBetterCutProb[h])))
        warm_MaxCutProb_Values.append([[p for i in range(len(warmstartMaxCutProb))], np.array(warmstartMaxCutProb)*100])
        warm_BetterCutProb_Values.append([[p for i in range(len(warmstartBetterCutProb))], np.array(warmstartBetterCutProb)*100])
        warm_means.append(np.median(warmstart))
        warm_value_list.append([[p for i in range(len(warmstart))], warmstart])
        if doCold:
            cold_MaxCutProb.append(np.median(coldstartMaxCutProb)*100)
            cold_BetterCutProb.append(np.median(coldstartBetterCutProb)*100)
            # save p, method, energy_median, maxcutchance_median, bettercutchance_median
            raw_median_results.append("{};{};{};{};{}".format(p, "cold", np.median(coldstart), np.median(coldstartMaxCutProb), np.median(coldstartBetterCutProb)))
            # cold_MaxCutProb_Values.append(coldstartMaxCutProb)
            for prob in coldstartMaxCutProb:
                cold_MaxCutProb_Values.append([p, prob*100])
            for prob in coldstartBetterCutProb:
                cold_BetterCutProb_Values.append([p, prob*100])
            cold_means.append(np.median(coldstart))
        print(warmstart)
        print(bestParamsForP)

    print([warm_means])
    print([warm_MaxCutProb])
    # print([warm_max, cold_max])


    #energygraph
    # warm_value_list = np.array(warm_value_list)


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

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    add_to_name = "_" + time +"_{}_{}".format(graph.shape[0], initialCut[1])
    path = os.getcwd() + "/results/" +  add_to_name
    print ("The current working directory is %s" % path)
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)


    rawResultsFile = open(path + "/rawAll"+add_to_name+".log", "w")
    rawResultsFile.write("\n".join(raw_all_results))
    rawResultsFile.close()


    rawResultsFile = open(path + "/rawMedian"+add_to_name+".log", "w")
    rawResultsFile.write("\n".join(raw_median_results))
    rawResultsFile.close()

    methodValues = [[] for methodCount in range(len(methods))]
    #probabilitygraph
    print(warm_MaxCutProb_Values)
    for p in range(len(p_range)):
        for methodCount in range(len(methods)):
            # [methodValues[methodCount][0].append(p) for i in range(len(warm_MaxCutProb_Values[p][1][methodCount]))]
            # [methodValues[methodCount][1].append(e) for e in warm_MaxCutProb_Values[p][1][methodCount]]
            [methodValues[methodCount].append([p_range[p],e]) for i,e in enumerate(warm_MaxCutProb_Values[p][1][methodCount])]


    methodValues = np.array(methodValues)
    colors = cm.get_cmap("rainbow", len(methods))

    if doCold == True:
        cold_MaxCutProb_Values = np.array(cold_MaxCutProb_Values)
        plt.scatter(cold_MaxCutProb_Values[:,0], cold_MaxCutProb_Values[:,1], marker=".", color='blue', label="Coldstarted", alpha=.4)
        plt.scatter(p_range, cold_MaxCutProb, linestyle="None", marker="x", color="b", alpha=.75)
    for methodCount, method in enumerate(methods):
        plt.scatter(methodValues[methodCount][:,0], methodValues[methodCount][:,1], marker=".", color = colors(methodCount), label=labels[methodCount] if labels else method, alpha=.4)
        plt.scatter(p_range, warm_MaxCutProb[methodCount], linestyle="None", marker="x",color = colors(methodCount), alpha=.8)
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("MaxCut Probability in %"), plt.title("MaxCut Probability")
    # plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(p_range)

    plt.savefig(path + "/compareMaxCutProbabilityMethods-"+add_to_name+".png", format="png")
    plt.show()
    plt.close()

    methodValues = [[] for methodCount in range(len(methods))]
    #probabilitygraph
    print(warm_BetterCutProb_Values)
    for p in range(len(p_range)):
        for methodCount in range(len(methods)):
            [methodValues[methodCount].append([p_range[p],e]) for i,e in enumerate(warm_BetterCutProb_Values[p][1][methodCount])]


    methodValues = np.array(methodValues)
    colors = cm.get_cmap("rainbow", len(methods))

    if doCold == True:
        cold_BetterCutProb_Values = np.array(cold_BetterCutProb_Values)
        plt.scatter(cold_BetterCutProb_Values[:,0], cold_BetterCutProb_Values[:,1], marker=".", color='blue', label="Coldstarted", alpha=.4)
        plt.scatter(p_range, cold_BetterCutProb, linestyle="None", marker="x", color="b", alpha=.75)

    for methodCount, method in enumerate(methods):
        plt.scatter(methodValues[methodCount][:,0], methodValues[methodCount][:,1], marker=".", color = colors(methodCount), label=labels[methodCount] if labels else method, alpha=.4)
        plt.scatter(p_range, warm_BetterCutProb[methodCount], linestyle="None", marker="x",color = colors(methodCount), alpha=.8)
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("BetterCut Probability in %"), plt.title("BetterCut Probability")
    # plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(p_range)

    plt.savefig(path + "/compareBetterCutProbabilityMethods-"+add_to_name+".png", format="png")
    plt.show()
    plt.close()








# graph = GraphGenerator.genMinimalGraph()
# cuts = bestGWcuts(graph, 10, 5, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
# GraphStorage.store("graphs/minimal-3v-3e-graph.txt", graph)
# GraphStorage.storeGWcuts("graphs/minimal-3v-3e-cuts.txt", cuts)


# graph = GraphGenerator.genButterflyGraph()
# graph = GraphGenerator.genGridGraph(4,4)
# graph = GraphGenerator.genFullyConnectedGraph(17)
# graph = GraphGenerator.genMustyGraph()
# graph = GraphGenerator.genRandomGraph(5,6)
# graph = GraphGenerator.genWarmstartPaperGraph()
# GraphPlotter.plotGraph(nx.Graph(graph))

# graph_loaded = GraphStorage.load("graphs/minimal-3v-3e-graph.txt")
# cuts_loaded = GraphStorage.loadGWcuts("graphs/minimal-3v-3e-cuts.txt")

# graph_loaded = GraphStorage.load("graphs/fullyConnected-6-paperversion-graph.txt")
# cuts_loaded = GraphStorage.loadGWcuts("graphs/fullyConnected-6-paperversion-cuts.txt")

graph_loaded = GraphStorage.load("graphs/prototype/fc-12-graph.txt")
cuts_loaded = GraphStorage.loadGWcuts("graphs/prototype/fc-12-cuts.txt")

# graph_loaded = GraphStorage.load("graphs/prototype/3r-12-graph.txt")
# cuts_loaded = GraphStorage.loadGWcuts("graphs/prototype/3r-12-cuts.txt")

# graph_loaded = GraphStorage.load("graphs/prototype/fc-24-graph.txt")
# cuts_loaded = GraphStorage.loadGWcuts("graphs/prototype/fc-24-cuts.txt")


# graph_loaded = GraphGenerator.genDiamondGraph()

print(cuts_loaded)

# Pick eta close to e^650/maxcut which results in e^eta*cut close to the maximum possible float
initial_cut = cuts_loaded[12]
eta= 650/(initial_cut[1]*1.2)
print(eta)

method_params = [None, (eta,)]
methods= [ None, "greedy"]
labels = [ r"$F_{EE}$", r"$F_{Greedy}$"]
# method_params = [None]
# methods= [None]
# labels = [ r"$F_{EE}$"]
# method_params = [ None, None]
# methods= [ None, "CVaR", "Gibbs", "Greedy", "ee-i"]
# method_params = [ None, (0.05,), (5,), None, None]
# labels = [ r"$F_{EE}$", r"$F_{0.05,CVar}$", r"$F_{5,Gibbs}$", r"$F_{Greedy}$", r"$F_{EE-I}$"]
knownMaxCut = np.array(cuts_loaded[-1][1])
epsilon = 0.125
doCold = True
doIncremental = True
onlyOptimizeCurrentP = True
useBestParams = False  #requires doIncremental = True
j = 20
p = [1,2,3]

compareWarmStartEnergyMethods(j, graph_loaded, p,  initialCut = initial_cut, knownMaxCut = knownMaxCut, epsilon=epsilon, methods=methods, method_params=method_params, doCold=doCold, doIncremental=doIncremental, onlyOptimizeCurrentP=onlyOptimizeCurrentP, labels=labels, useBestParmas=useBestParams)







# print(params.x)
# plotCircuit(graph, params.x, p, FakeYorktown())
# plotSolution(graph, params.x, p)
# plt.plot(costs_history)
# plt.show()
