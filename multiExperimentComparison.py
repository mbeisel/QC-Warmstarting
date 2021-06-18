from datetime import datetime

from graphStorage import GraphStorage
from qaoaComparison import compareWarmStartEnergyMethods
import numpy as np

#{key: name, value: cutarrayid}
three_reg_graphs = {}
fully_conn_graphs = {}
random_graphs = {}


def comparisonForMultipleGraphs(graphs, graphtype ='Not-specified', optimizer ='Cobyla', do_cold=True, do_incremental=True, only_optimize_current_p=True):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    foldername = time + 'multicomparison-' + graphtype + '/'
    for g in graphs:
        graph_loaded = GraphStorage.load("graphs/prototype/multiExperiment/{}-graph.txt".format(g))
        cuts_loaded = GraphStorage.loadGWcuts("graphs/prototype/multiExperiment/{}-cuts.txt".format(g))
        print(cuts_loaded)

        initial_cut = cuts_loaded[graphs[g]]
        methods = [None, "CVaR", "Gibbs", "Greedy", "ee-i"]
        method_params = [None, (0.05,), (5,), None, None]
        labels = [r"$F_{EE}$", r"$F_{0.05,CVar}$", r"$F_{5,Gibbs}$", r"$F_{Greedy}$", r"$F_{EE-I}$"]
        epsilon = 0.15
        known_max_cut = np.array(cuts_loaded[-1][1])
        do_cold = do_cold
        do_incremental = do_incremental
        only_optimize_current_p = only_optimize_current_p
        use_best_params = False  #requires doIncremental = True
        optimize_epsilon = False
        j = 20
        p = [1,2,3]
        optimizer = optimizer
        folder_name_final = foldername + g

        compareWarmStartEnergyMethods(j, graph_loaded, p, initial_cut= initial_cut, known_max_cut= known_max_cut, epsilon=epsilon, methods=methods, method_params=method_params, do_cold=do_cold, do_incremental=do_incremental, only_optimize_current_p=only_optimize_current_p, labels=labels, use_best_parmas=use_best_params, optimize_epsilon=optimize_epsilon, optimizer=optimizer, foldername=folder_name_final)




# incrementive Comparison
comparisonForMultipleGraphs(fully_conn_graphs, 'fc', do_incremental=True, only_optimize_current_p=True)


#optimizerComparison
optimizers = ['Cobyla', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC','SLSQP']
comparisonForMultipleGraphs(fully_conn_graphs, 'fc', optimizer=optimizers[0])



