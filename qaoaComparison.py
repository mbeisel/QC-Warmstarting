import os


from matplotlib import cm
from graphGenerator import GraphGenerator
from graphStorage import GraphStorage
from copy import copy
from maxcutQaoa import objectiveFunction, objectiveFunctionBest, cut_costs
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from MinimizeWrapper import MinimizeWrapper
from pathlib import Path




def compareWarmStartEnergyMethods(iterations, graph, p_range, initial_cut, known_max_cut = None, only_optimize_current_p = False, epsilon =0.25, do_cold = False, methods = None, do_incremental=True, method_params = None, labels = None, use_best_parmas = False, optimize_epsilon=False, optimizer='Cobyla', foldername=None, hamming_distance =None):
    #clear cutcost list from maxcutQaoa
    cut_costs.clear()

    if(initial_cut):
        print("Use: {} with epsilon: {}".format(initial_cut, epsilon))
    warm_means = []
    warm_value_list = []
    warm_max_cut_prob = [[] for i in range(len(methods))]
    warm_max_cut_prob_values = []
    warm_better_cut_prob = [[] for i in range(len(methods))]
    warm_better_cut_prob_values = []
    cold_means = []
    cold_max_cut_prob = []
    cold_max_cut_prob_values = []
    cold_better_cut_prob = []
    cold_better_cut_prob_values = []
    best_params_for_pcold = [[-999999999,None] for i in range(len(p_range))]
    optimizer_options = None
    raw_median_results = ["doincremental:{}; onlyOptimizeCurrentP:{}; useBestParams:{}; epsilon:{}; initialCut:{}; Hamming_distance:{}; optimizer: {}".format(do_incremental, only_optimize_current_p, use_best_params, epsilon, initial_cut, hamming_distance or -1, optimizer)]
    raw_all_results = ["doincremental:{}; onlyOptimizeCurrentP:{}; useBestParams:{}; epsilon:{}; initialCut:{}; Hamming_distance: {}; optimizer: {}".format(do_incremental, only_optimize_current_p, use_best_params, epsilon, initial_cut, hamming_distance or -1, optimizer)]
    warm_all_method_params = [[[] for i in range(iterations)] for i in range(len(methods))]
    cold_all_params = [[] for i in range(iterations)]



    p_range = list(p_range)
    best_params_for_p = [[[-999999999,None] for i in range(len(p_range))] for i in range(len(methods))]
    for count,p in enumerate(p_range):
        warmstart, warmstart_max_cut_prob, warmstart_better_cut_prob = [[] for i in range(len(methods))], [[] for i in range(len(methods))], [[] for i in range(len(methods))]
        coldstart, coldstart_max_cut_prob, coldstart_better_cut_prob = [], [], []

        optimizer_steps_list, runtime_list = [[] for i in range(len(methods))], [[] for i in range(len(methods))]
        optimizer_steps_list_cold, runtime_list_cold = [], []

        for i in range(0, 1):
            #optimize j times starting with different startvalues
            for j in range(iterations):

                #################
                #   Coldstart   #
                #################

                params_raw = np.random.default_rng().uniform(0, np.pi, size=p*2)
                if do_incremental and p > 1:
                    params_raw = np.concatenate((np.random.default_rng().uniform(0, np.pi, size=2),np.zeros(2*(p-1))))
                    # params_raw = np.zeros(2*p)
                if do_cold ==True:
                    params_cold = copy(params_raw)

                    if( p > 1 and do_incremental and best_params_for_pcold[count - 1][0] != -999999999):
                        if not use_best_parmas:
                            for e in range(p_range[count-1]*2):
                                params_cold[e] = cold_all_params[j][e]
                        else:
                            for e in range(p_range[count-1]*2):
                                params_cold[e] = best_params_for_pcold[count-1][1][e]
                    if only_optimize_current_p == True:
                        params_cold_optimized = MinimizeWrapper().minimize(objectiveFunction, params_cold[p_range[count-1]*2:] if p > 1 else params_cold, method=optimizer,
                                                         args=(None,graph, None, p, list(params_cold[:p_range[count-1]*2]) if p > 1 else None), options=optimizer_options)
                        if p > 1:
                            params_cold_optimized.bestValue[0] = list(params_cold[:p_range[count-1]*2]) + list(params_cold_optimized.bestValue[0])
                    else:
                        params_cold_optimized = MinimizeWrapper().minimize(objectiveFunction, params_cold, method=optimizer, args=(None,graph, None, p),
                                                                 options=optimizer_options)
                    energy_cold, cut_cold, max_cut_chance_cold, better_cut_chance_cold = objectiveFunctionBest(params_cold_optimized.bestValue[0], None, graph, None, p,
                                                                                                       knownMaxCut= known_max_cut,
                                                                                                       showHistogram=False)
                    if best_params_for_pcold[count][0] < energy_cold:
                        best_params_for_pcold[count][0] = energy_cold
                        best_params_for_pcold[count][1] = list(params_cold_optimized.bestValue[0])
                    if not use_best_parmas:
                        cold_all_params[j] = list(params_cold_optimized.bestValue[0])
                    coldstart.append(energy_cold)
                    coldstart_max_cut_prob.append(max_cut_chance_cold)
                    coldstart_better_cut_prob.append(better_cut_chance_cold)
                    optimizer_steps_list_cold.append(len(params_cold_optimized.optimizationPath))
                    runtime_list_cold.append(params_cold_optimized.optimizationTime)
                    print("maxcutchance for coldstart:{} at j={}".format(max_cut_chance_cold, j))
                    raw_all_results.append("{};{},{};{};{};{};{};{};{}".format(p,j, "cold", energy_cold, max_cut_chance_cold, 0, ','.join(list(str(e) for e in params_cold_optimized.bestValue[0])),optimizer_steps_list_cold[j], runtime_list_cold[j]))


                #################
                #   Warmstart   #
                #################

                energy_warm_list, cut_warm_list, max_cut_chance_warm_list, better_cut_chance_warm_list, params_warm_list = [[] for i in range(len(methods))], [[] for i in range(len(methods))], [[] for i in range(len(methods))], [[] for i in range(len(methods))],[[] for i in range(len(methods))]

                if type(epsilon) is float or type(epsilon) == int:
                    epsilon = [epsilon] * len(methods)
                for method_count, method in enumerate(methods):
                    # bestCut = epsilonFunction(initialCut[0], epsilon=epsilon[methodCount])
                    params = copy(params_raw)

                    if(p > 1 and do_incremental and best_params_for_p[method_count][count - 1][0] != -999999999):
                        if not use_best_parmas:
                            for e in range(p_range[count-1]*2):
                                params[e] = warm_all_method_params[method_count][j][e]
                        else:
                            for e in range(p_range[count-1]*2):
                                params[e] = best_params_for_p[method_count][count-1][1][e]


                    #optimize k times with the same startvalues and take the best
                    for k in range(1):
                        cons = []
                        cons.append({'type': 'ineq', 'fun': lambda x: x[-1] - 0})
                        cons.append({'type': 'ineq', 'fun': lambda x: 0.5 - x[-1]})

                        if optimize_epsilon == True:
                            if only_optimize_current_p == True:
                                optimization_params = np.append(params[p_range[count-1]*2:] if p > 1 else params, epsilon[method_count])
                                params_warm_optimized = MinimizeWrapper().minimize(objectiveFunction, optimization_params, method=optimizer, constraints=cons,
                                                                                   args=(None, graph, initial_cut[0], p, list(params[:p_range[count - 1] * 2]) if p > 1 else None, initial_cut[1], method, method_params[method_count]), options=optimizer_options)
                                if p > 1:
                                    params_warm_optimized.bestValue[0] = list(params[:p_range[count-1]*2]) + list(params_warm_optimized.bestValue[0])
                            else:
                                optimization_params = np.append(params, epsilon[method_count])
                                params_warm_optimized = MinimizeWrapper().minimize(objectiveFunction, optimization_params, method=optimizer, constraints=cons,
                                                                                   args=(None, graph, initial_cut[0], p, None, initial_cut[1], method, method_params[method_count]), options=optimizer_options)
                            energy_warm, cut_warm, max_cut_chance_warm, better_cut_chance_warm = objectiveFunctionBest(params_warm_optimized.bestValue[0], None, graph, initial_cut[0], p,
                                                                                                               knownMaxCut= known_max_cut,
                                                                                                               showHistogram=False, inputCut=initial_cut[1], method=method, method_params=method_params[method_count])
                            print(params_warm_optimized.bestValue[0][-1])
                        else:
                            if only_optimize_current_p == True:
                                params_warm_optimized = MinimizeWrapper().minimize(objectiveFunction, params[p_range[count-1]*2:] if p > 1 else params, method=optimizer,
                                                                                   args=(epsilon[method_count], graph, initial_cut[0], p, list(params[:p_range[count - 1] * 2]) if p > 1 else None, initial_cut[1], method, method_params[method_count]), options=optimizer_options)
                                if p > 1:
                                    params_warm_optimized.bestValue[0] = list(params[:p_range[count-1]*2]) + list(params_warm_optimized.bestValue[0])
                            else:
                                params_warm_optimized = MinimizeWrapper().minimize(objectiveFunction, params, method=optimizer,
                                                                                   args=(epsilon[method_count], graph, initial_cut[0], p, None, initial_cut[1], method, method_params[method_count]), options=optimizer_options)

                            energy_warm, cut_warm, max_cut_chance_warm, better_cut_chance_warm = objectiveFunctionBest(params_warm_optimized.bestValue[0], epsilon[method_count], graph, initial_cut[0], p,
                                                                                                               knownMaxCut= known_max_cut,
                                                                                                               showHistogram=False, inputCut=initial_cut[1], method=method, method_params=method_params[method_count])
                        if best_params_for_p[method_count][count][0] < energy_warm:
                            best_params_for_p[method_count][count][0] = energy_warm
                            best_params_for_p[method_count][count][1] = list(params_warm_optimized.bestValue[0])
                        if not use_best_parmas:
                            warm_all_method_params[method_count][j] = list(params_warm_optimized.bestValue[0])
                        energy_warm_list[method_count].append(energy_warm)
                        cut_warm_list[method_count].append(cut_warm)
                        max_cut_chance_warm_list[method_count].append(max_cut_chance_warm)
                        better_cut_chance_warm_list[method_count].append(better_cut_chance_warm)
                        params_warm_list[method_count].append(list(params_warm_optimized.bestValue[0]))
                        optimizer_steps_list[method_count].append(len(params_warm_optimized.optimizationPath))
                        runtime_list[method_count].append(params_warm_optimized.optimizationTime)
                    warmstart[method_count].append(np.max(energy_warm_list[method_count]))
                    warmstart_max_cut_prob[method_count].append(np.max(max_cut_chance_warm_list[method_count]))
                    warmstart_better_cut_prob[method_count].append(np.max(better_cut_chance_warm_list[method_count]))
                    print("maxcutchance for method {}:{} at j={}".format(method, np.max(max_cut_chance_warm_list[method_count]), j))
                    raw_all_results.append("{};{};{};{};{};{};{};{};{}".format(p,j, method, np.max(energy_warm_list[method_count]), np.max(max_cut_chance_warm_list[method_count]), np.max(better_cut_chance_warm_list[method_count]), ','.join(list(str(e) for e in params_warm_list[method_count][0])), optimizer_steps_list[method_count][j],runtime_list[method_count][j]))


            print("{:.2f}%".format(100 * ((count+1)/len(p_range))))

        print("WARMSTARTPROB")
        print(warmstart_max_cut_prob)
        for h, method in enumerate(methods):
            warm_max_cut_prob[h].append(np.median(warmstart_max_cut_prob[h])*100)
            warm_better_cut_prob[h].append(np.median(warmstart_better_cut_prob[h])*100)
            # save p, method, energy_median, maxcutchance_median, bettercutchance_median
            raw_median_results.append("{};{};{};{};{};{};{}".format(p, methods[h], np.median(warmstart[h]), np.median(warmstart_max_cut_prob[h]), np.median(warmstart_better_cut_prob[h]), np.mean(optimizer_steps_list[h]),np.mean(runtime_list[h])))
        warm_max_cut_prob_values.append([[p for i in range(len(warmstart_max_cut_prob))], np.array(warmstart_max_cut_prob)*100])
        warm_better_cut_prob_values.append([[p for i in range(len(warmstart_better_cut_prob))], np.array(warmstart_better_cut_prob)*100])
        warm_means.append(np.median(warmstart))
        warm_value_list.append([[p for i in range(len(warmstart))], warmstart])
        if do_cold:
            cold_max_cut_prob.append(np.median(coldstart_max_cut_prob)*100)
            cold_better_cut_prob.append(np.median(coldstart_better_cut_prob)*100)
            # save p, method, energy_median, maxcutchance_median, bettercutchance_median
            raw_median_results.append("{};{};{};{};{};{};{}".format(p, "cold", np.median(coldstart), np.median(coldstart_max_cut_prob), np.median(coldstart_better_cut_prob), np.mean(optimizer_steps_list_cold), np.mean(runtime_list_cold)))
            # cold_MaxCutProb_Values.append(coldstartMaxCutProb)
            for prob in coldstart_max_cut_prob:
                cold_max_cut_prob_values.append([p, prob*100])
            for prob in coldstart_better_cut_prob:
                cold_better_cut_prob_values.append([p, prob*100])
            cold_means.append(np.median(coldstart))
        print(warmstart)
        print(best_params_for_p)

    print([warm_means])
    print([warm_max_cut_prob])


    #################
    #    Logging    #
    #################
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    add_to_name = "_" + time +"_{}_{}".format(graph.shape[0], initial_cut[1])
    path = os.getcwd() + "/results/" + foldername if foldername else os.getcwd() + "/results/" + add_to_name


    print("The current working directory is %s" % path)
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)


    raw_results_file = open(path + "/rawAll"+add_to_name+".log", "w")
    raw_results_file.write("\n".join(raw_all_results))
    raw_results_file.close()


    raw_results_file = open(path + "/rawMedian"+add_to_name+".log", "w")
    raw_results_file.write("\n".join(raw_median_results))
    raw_results_file.close()


    #################
    # Graphplotting #
    #################
    method_values = [[] for method_count in range(len(methods))]
    #probabilitygraph
    print(warm_max_cut_prob_values)
    for p in range(len(p_range)):
        for method_count in range(len(methods)):
            [method_values[method_count].append([p_range[p],e]) for i,e in enumerate(warm_max_cut_prob_values[p][1][method_count])]


    method_values = np.array(method_values)
    colors = cm.get_cmap("rainbow", len(methods))

    if do_cold == True:
        cold_max_cut_prob_values = np.array(cold_max_cut_prob_values)
        plt.scatter(cold_max_cut_prob_values[:,0], cold_max_cut_prob_values[:, 1], marker=".", color='blue', label="Coldstarted", alpha=.4)
        plt.scatter(p_range, cold_max_cut_prob, linestyle="None", marker="x", color="b", alpha=.75)
    for method_count, method in enumerate(methods):
        plt.scatter(method_values[method_count][:,0], method_values[method_count][:, 1], marker=".", color = colors(method_count), label=labels[method_count] if labels else method, alpha=.4)
        plt.scatter(p_range, warm_max_cut_prob[method_count], linestyle="None", marker="x",color = colors(method_count), alpha=.8)
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("MaxCut Probability in %"), plt.title("MaxCut Probability")
    # plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(p_range)

    plt.savefig(path + "/compareMaxCutProbabilityMethods-"+add_to_name+".png", format="png")
    # plt.show()
    plt.close()

    method_values = [[] for method_count in range(len(methods))]
    #probabilitygraph
    print(warm_better_cut_prob_values)
    for p in range(len(p_range)):
        for method_count in range(len(methods)):
            [method_values[method_count].append([p_range[p],e]) for i,e in enumerate(warm_better_cut_prob_values[p][1][method_count])]


    method_values = np.array(method_values)
    colors = cm.get_cmap("rainbow", len(methods))

    if do_cold == True:
        cold_better_cut_prob_values = np.array(cold_better_cut_prob_values)
        plt.scatter(cold_better_cut_prob_values[:, 0], cold_better_cut_prob_values[:,1], marker=".", color='blue', label="Coldstarted", alpha=.4)
        plt.scatter(p_range, cold_better_cut_prob, linestyle="None", marker="x", color="b", alpha=.75)

    for method_count, method in enumerate(methods):
        plt.scatter(method_values[method_count][:, 0], method_values[method_count][:,1], marker=".", color = colors(method_count), label=labels[method_count] if labels else method, alpha=.4)
        plt.scatter(p_range, warm_better_cut_prob[method_count], linestyle="None", marker="x",color = colors(method_count), alpha=.8)
    plt.legend(loc="best"), plt.xlabel("p"), plt.ylabel("BetterCut Probability in %"), plt.title("BetterCut Probability")
    # plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(p_range)

    plt.savefig(path + "/compareBetterCutProbabilityMethods-"+add_to_name+".png", format="png")
    # plt.show()
    plt.close()

    return raw_median_results, raw_all_results





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

# print(cuts_loaded)

# Pick eta close to e^650/maxcut which results in e^eta*cut close to the maximum possible float
initial_cut = cuts_loaded[12]
eta= 650/(initial_cut[1]*1.2)
# print(eta)

# method_params = [None, None]
# methods= [ None, "greedy"]
# labels = [ r"$F_{EE}$", r"$F_{Greedy}$"]
# epsilon = [0.15, 0.025]
# epsilon = [0.15, 0.15]
# method_params = [None]
# methods= [None]
# labels = [ r"$F_{EE}$"]
# method_params = [ None, None]
methods = [None, "CVaR", "Gibbs", "Greedy", "ee-i"]
method_params = [None, (0.05,), (5,), None, None]
labels = [r"$F_{EE}$", r"$F_{0.05,CVar}$", r"$F_{5,Gibbs}$", r"$F_{Greedy}$", r"$F_{EE-I}$"]
epsilon = 0.15
known_max_cut = np.array(cuts_loaded[-1][1])
do_cold = True
do_incremental = True
only_optimize_current_p = True
use_best_params = False  #requires doIncremental = True
optimize_epsilon = False
j = 10
p = [1,2,3]
optimizer = 'Cobyla'
hamming_distance = None

# compareWarmStartEnergyMethods(j, graph_loaded, p, initial_cut= initial_cut, known_max_cut= known_max_cut, epsilon=epsilon, methods=methods, method_params=method_params, do_cold=do_cold, do_incremental=do_incremental, only_optimize_current_p=only_optimize_current_p, labels=labels, use_best_parmas=use_best_params, optimize_epsilon=optimize_epsilon, optimizer=optimizer, hamming_distance=hamming_distance)







# print(params.x)
# plotCircuit(graph, params.x, p, FakeYorktown())
# plotSolution(graph, params.x, p)
# plt.plot(costs_history)
# plt.show()
