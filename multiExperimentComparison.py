import os
from pathlib import Path
from datetime import datetime

from graphStorage import GraphStorage
from qaoaComparison import compareWarmStartEnergyMethods
import numpy as np
import json

#[graphpath, cutpath, ~0.878 cutarrayindex, hamming distance]
path = "graphs/prototype/multiExperiment"
fname = f"{path}/initial_cuts-12.txt"
a_file = open(fname, "r")
all_12_graphs = json.loads(a_file.read())

fname = f"{path}/fc-initial_cuts-12.txt"
a_file = open(fname, "r")
fc_12_graphs = json.loads(a_file.read())

fname = f"{path}/3r-initial_cuts-12.txt"
a_file = open(fname, "r")
three_reg_12_graphs = json.loads(a_file.read())

fname = f"{path}/rand-initial_cuts-12.txt"
a_file = open(fname, "r")
rand_12_graphs = json.loads(a_file.read())

fname = f"graphs/prototype/initial_cuts_mixed.txt"
a_file = open(fname, "r")
graph_mix = json.loads(a_file.read())

def comparisonForMultipleGraphs(graphs, folder_add_to_name ='Not-specified', optimizer ='Cobyla', do_cold=True, do_incremental=True, only_optimize_current_p=True):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    foldername = time + 'multicomparison-' + folder_add_to_name
    path = os.getcwd() + "/results/" + foldername
    results = []
    all_results = []
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    for i,graph in enumerate(graphs):
        graph_loaded = GraphStorage.load(graph[0])
        cuts_loaded = GraphStorage.loadGWcuts(graph[1])

        initial_cut = cuts_loaded[graph[2]]
        method_params = [None, (0.05,), (5,) if len(initial_cut[0]) == 12 else (2,), None, None]
        methods= [ None, "CVaR", "Gibbs", "greedy", "EE-I"]
        labels = [ r"$F_{EE}$", r"$F_{Greedy}$"]
        labels = [r"$F_{EE}$", r"$F_{0.05,CVar}$", r"$F_{5,Gibbs}$" if method_params[2][0] == 5 else r"$F_{2,Gibbs}$", r"$F_{Greedy}$", r"$F_{EE-I}$"]
        # methods = [None, "CVaR", "Gibbs", "Greedy", "ee-i"]
        # method_params = [None, (0.05,), (5,), None, None]
        # labels = [r"$F_{EE}$", r"$F_{0.05,CVar}$", r"$F_{5,Gibbs}$", r"$F_{Greedy}$", r"$F_{EE-I}$"]
        epsilon = 0.15
        known_max_cut = np.array(cuts_loaded[-1][1])
        do_cold = do_cold
        do_incremental = do_incremental
        only_optimize_current_p = only_optimize_current_p
        use_best_params = False  #requires doIncremental = True
        optimize_epsilon = True
        only_optimize_epsilon_at_p1 = True
        j = 20
        p = [0,1,2,3]
        optimizer = optimizer
        folder_name_final = foldername + '/' + str(i)

        medianResults, rawAllResults = compareWarmStartEnergyMethods(j, graph_loaded, p, initial_cut= initial_cut, known_max_cut= known_max_cut, epsilon=epsilon, methods=methods, method_params=method_params, do_cold=do_cold, do_incremental=do_incremental, only_optimize_current_p=only_optimize_current_p, labels=labels, use_best_parmas=use_best_params, optimize_epsilon=optimize_epsilon, optimizer=optimizer, foldername=folder_name_final, only_optimize_epsilon_at_p1=only_optimize_epsilon_at_p1)

        [results.append(result) for result in medianResults]
        [all_results.append(result) for result in rawAllResults]

        # if i >= 2:
        #     break

    raw_results_file = open(path + "/rawMedian" + ".log", "w")
    raw_results_file.write("\n".join(results))
    raw_results_file.close()

    raw_results_file = open(path + "/rawAll" + ".log", "w")
    raw_results_file.write("\n".join(all_results))
    raw_results_file.close()

# incrementive Comparison
print(graph_mix)
comparisonForMultipleGraphs(graph_mix, 'mixed', do_cold=False, do_incremental=True, only_optimize_current_p=True)


#optimizerComparison
optimizers = ['Cobyla', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC','SLSQP']
for i in range(1):
    optimizer = optimizers[i]
    # comparisonForMultipleGraphs(fc_12_graphs, folder_add_to_name=optimizer, optimizer=optimizer)



