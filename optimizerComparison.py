from graphGenerator import GraphGenerator
from helperFunctions import epsilonFunction
from goemansWilliamson import bestGWcuts
from copy import deepcopy
from scipy.optimize import minimize
from maxcutQaoa import objectiveFunction, objectiveFunctionBest
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time


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


# graph = GraphGenerator.genButterflyGraph()
# graph = GraphGenerator.genGridGraph(4,4)
# graph = GraphGenerator.genFullyConnectedGraph(17)
# graph = GraphGenerator.genMustyGraph()
# graph = GraphGenerator.genRandomGraph(5,6)
graph = GraphGenerator.genWarmstartPaperGraph()
# GraphPlotter.plotGraph(nx.Graph(graph))


compareOptimizerEnergy(graph, [1,2,3], ["Cobyla", "TNC"])
# compareOptimizerEnergy(graph, [1,2], ["Cobyla", "Powell"])