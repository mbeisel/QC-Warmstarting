from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from qiskit import transpile

from goemansWilliamson import bestGWcuts
from graphGenerator import GraphGenerator
from graphStorage import GraphStorage
from maxcutQaoa import objectiveFunction, objectiveFunctionBest, cost_function_C
from scipy.optimize import minimize
from helperFunctions import epsilonFunction
from datetime import datetime
from QAOACircuitGenerator import QAOACircuitGenerator

# Gridsearch for p = 1
def gridSearch(objective_fun, Graph, approximation_List, cut_size, maxCut, p, gamma_step_size=0.5, gammaStart=0, gammaEnd=np.pi,
               beta_step_size=0.5, betaStart=0, betaEnd=np.pi, plot=False, method=None, fname=None):

    a_gamma = np.arange(gammaStart, gammaEnd, gamma_step_size)
    a_beta = np.arange(betaStart, betaEnd, beta_step_size)

    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)
    shape = a_gamma.shape
    a_gamma, a_beta = a_gamma.flatten(), a_beta.flatten()

    # energy, _, _ = objectiveFunctionBest([0, np.pi/2], Graph, approximation_List, p)
    # print(energy)
    F1 = []
    print("Total Operations: {}".format(len(a_gamma)))
    for i in range(len(a_gamma)):
        if(p > 1):
            # optimizer_options = ({"rhobeg": np.pi/2})
            optimizer_options = None
            params = np.random.uniform(0, 6*np.pi, size=2*(p-1))
            print("test")
            # params = [0,0] * (p-1)
            energyList = []
            for j in range(1):
                optimizedparams = minimize(objectiveFunction, params, method="COBYLA",
                                                 args=(Graph, approximation_List, p, [a_gamma[i], a_beta[i]]), options=optimizer_options)
                optimizedparams = [a_gamma[i], a_beta[i]] + list(optimizedparams.x)

                energy, bestCut, maxCutChance = objectiveFunctionBest(optimizedparams, Graph, approximation_List, p, 27)
                energyList.append(energy)
                print("opt{}, energy {} bestCut {}, probability {}".format(optimizedparams,energy, bestCut, maxCutChance ))

                # TODO MAX MIN DEPENDENT ON Positive / Negative Values
            print(energyList)
            F1.append(np.max(energyList))
        else:
            F1.append(objective_fun([a_gamma[i], a_beta[i]], Graph, approximation_List, p, inputCut=cut_size, method=method, maxCut=maxCut))
        print("Current Step: {}".format(i+1)) if (i + 1) % 10 == 0 else None

    F1 = np.array(F1)

    # Grid search for the minimizing variables
    result = np.where(F1 == np.amin(F1))
    gamma, beta = a_gamma[result[0][0]], a_beta[result[0][0]]

    # Plot the expetation value F1
    if plot or fname:
        fig = plt.figure()
        #ax  = fig.gca(projection='3d')

        # size = len(np.arange(0, amountOfSteps, step_size))
        a_gamma, a_beta, F1 = a_gamma.reshape(shape), a_beta.reshape(shape), F1.reshape(shape)
        #surf = ax.plot_surface(a_gamma, a_beta, F1, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        #ax.set_zlim(np.amin(F1)-1,np.amax(F1)+1)
        #ax.zaxis.set_major_locator(LinearLocator(5))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax = fig.add_subplot(1,1,1)
        img = ax.contourf(a_gamma, a_beta, F1, cmap=cm.get_cmap('viridis', 256), antialiased=True)
        # img = ax.pcolormesh(a_gamma, a_beta, F1, cmap=cm.get_cmap('viridis', 256), rasterized = True)
        ax.scatter(gamma, beta, s=100, edgecolor="r", facecolor="none", linewidth=3.5)

        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.set_xticks(np.arange(0, 2*np.pi+0.01, np.pi/2))
        labels = ['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
        ax.set_xticklabels(labels)

        ax.set_yticks(np.arange(0, np.pi+0.01, np.pi/2))
        labels = ['$0$', r'$\pi/2$', r'$\pi$']
        ax.set_yticklabels(labels)

        cbar = fig.colorbar(img)
        cbar.ax.set_ylabel("Energy")
    if (fname):
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()


    return [gamma, beta], np.amin(F1)

# from graphGenerator import GraphGenerator
# graph = GraphGenerator.genFullyConnectedGraph(20)
# cuts = bestGWcuts(graph, 10, 5, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
# GraphStorage.store("graphs/fullyConnected-20-graph.txt", graph)
# GraphStorage.storeGWcuts("graphs/fullyConnected-20-cuts.txt", cuts)

graph_loaded = GraphStorage.load("graphs/grid-3-4-graph.txt")
cuts_loaded = GraphStorage.loadGWcuts("graphs/grid-3-4-cuts.txt")

# graph_loaded = GraphStorage.load("graphs/fullyConnected-12-graph.txt")
# cuts_loaded = GraphStorage.loadGWcuts("graphs/fullyConnected-12-cuts.txt")
epsilon = 0.25
cuts_loaded = np.array([[epsilonFunction(cut[0], epsilon), cut[1]] for cut in cuts_loaded], dtype=object)

# graph_loaded = GraphGenerator.genDiamondGraph()
print(cuts_loaded)
cut_used = cuts_loaded[1]
maxcut = cuts_loaded[-1,1]
methods = ["max", 0, 1, 2, 3]

#cut_used = cuts_loaded[2,0]
print("Selected GWCut {}; Maxcut: {}".format(cut_used, maxcut))

for method in methods:
    filename = "results/Gridsearch-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")+str(len(cut_used[0]))+"-"+str(cut_used[1])+"-method"+str(method)+"-eps"+str(epsilon)+".png"
    params, min_energy = gridSearch(
        objectiveFunction,
        graph_loaded, cut_used[0], cut_used[1], maxcut,
        1,
        gamma_step_size=np.pi/10, gammaStart=-np.pi*0, gammaEnd=np.pi*2,
        betaEnd=np.pi, beta_step_size=np.pi/10,
        method=method, plot=True, fname=filename)

    print("Params: {}, Min energy: {}".format(params, min_energy))
