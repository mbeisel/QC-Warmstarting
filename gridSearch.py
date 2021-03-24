import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from goemansWilliamson import bestGWcuts
from graphGenerator import GraphGenerator
from maxcutQaoa import objectiveFunction, objectiveFunctionBest, cost_function_C

# Gridsearch for p = 1
def gridSearch(objective_fun, Graph, approximation_List, p, step_size=0.2, plot=False, fname=None):
    a_gamma = np.arange(0, np.pi, step_size)
    a_beta = np.arange(0, np.pi, step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)
    stepAmount = a_gamma.shape
    a_gamma, a_beta = a_gamma.flatten(), a_beta.flatten()

    energy, _, _ = objectiveFunctionBest([0, np.pi/2], Graph, approximation_List, p)
    print(energy)
    F1 = []
    print("Total Operations: {}".format(len(a_gamma)))
    for i in range(len(a_gamma)):
        F1.append(objective_fun([a_gamma[i], a_beta[i]], Graph, approximation_List, p))
        print("Current Step: {}".format(i)) if i % 10 == 0 else None
    F1 = np.array(F1)

    # Grid search for the minimizing variables
    result = np.where(F1 == np.amin(F1))
    gamma, beta = a_gamma[result[0][0]], a_beta[result[0][0]]

    # Plot the expetation value F1
    if plot or fname:
        fig = plt.figure()
        #ax  = fig.gca(projection='3d')

        # size = len(np.arange(0, amountOfSteps, step_size))
        a_gamma, a_beta, F1 = a_gamma.reshape(stepAmount), a_beta.reshape(stepAmount), F1.reshape(stepAmount)
        #surf = ax.plot_surface(a_gamma, a_beta, F1, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        #ax.set_zlim(np.amin(F1)-1,np.amax(F1)+1)
        #ax.zaxis.set_major_locator(LinearLocator(5))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax = fig.add_subplot(1,1,1)
        img = ax.contourf(a_gamma, a_beta, F1, cmap=cm.coolwarm, antialiased=True)
        cbar = fig.colorbar(img)
        cbar.ax.set_ylabel("Energy")
    if (fname):
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

    return np.array([gamma, beta]), np.amin(F1)

graph = GraphGenerator.genFullyConnectedGraph(20)
RawBestCuts = bestGWcuts(graph, 10, 2, continuous=False, epsilon=0.25, cost_fun=cost_function_C)
print(RawBestCuts)
print("Selected GWCut {}".format(RawBestCuts[0]))
gridSearch(objectiveFunction, graph, RawBestCuts[0,0], 1, plot=True )