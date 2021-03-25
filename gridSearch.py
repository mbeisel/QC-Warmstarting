import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from goemansWilliamson import bestGWcuts
from graphGenerator import GraphGenerator
from maxcutQaoa import objectiveFunction, objectiveFunctionBest, cost_function_C
from scipy.optimize import minimize


# Gridsearch for p = 1
def gridSearch(objective_fun, Graph, approximation_List, p, gamma_step_size=0.5, gammaStart = 0, gammaEnd = np.pi,
               beta_step_size = 0.5, betaStart =0, betaEnd=np.pi, plot=False, fname=None):

    a_gamma = np.arange(gammaStart, gammaEnd, gamma_step_size)
    a_beta = np.arange(betaStart, betaEnd, beta_step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)
    stepAmount = a_gamma.shape
    a_gamma, a_beta = a_gamma.flatten(), a_beta.flatten()

    # energy, _, _ = objectiveFunctionBest([0, np.pi/2], Graph, approximation_List, p)
    # print(energy)
    F1 = []
    print("Total Operations: {}".format(len(a_gamma)))
    for i in range(len(a_gamma)):
        if(p > 1):
            optimizer_options = ({"rhobeg": np.pi/2})
            params = np.random.uniform(0, np.pi, size=2*(p-1))
            optimizedparams = minimize(objectiveFunction, params, method="COBYLA",
                                             args=(graph, approximation_List, p, [a_gamma[i], a_beta[i]]), options=optimizer_options)
            optimizedparams = [a_gamma[i], a_beta[i]] + list(optimizedparams.x)
            print("opt{}".format(optimizedparams))
            F1.append(objective_fun(optimizedparams, Graph, approximation_List, p))
        else:
            F1.append(objective_fun([a_gamma[i], a_beta[i]], Graph, approximation_List, p))
        print("Current Step: {}".format(i)) if i +1 % 10 == 0 else None
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

graph = GraphGenerator.genFullyConnectedGraph(11)
RawBestCuts = bestGWcuts(graph, 10, 2, continuous=False, epsilon=0.25, cost_fun=cost_function_C)
print(RawBestCuts)
print("Selected GWCut {}".format(RawBestCuts[0]))


# Default Gridsearch for p = 1
# gridSearch(objectiveFunction, graph, RawBestCuts[0,0], 1, plot=True )

#p=1 detailed search around 0, pi/2
# gridSearch(objectiveFunction, graph, RawBestCuts[0,0], 1, gamma_step_size=0.0002, gammaStart=-0.001, gammaEnd=0.001, beta_step_size=0.001,betaStart=np.pi/2 -0.01, betaEnd=np.pi/2 +0.01, plot=True )

# Gridsearch for p=3 over the entire grid
gridSearch(objectiveFunction, graph, RawBestCuts[0,0], 3, gamma_step_size=1, beta_step_size=1 ,plot=True )

#Gridsearch for p=3 from -3.4 to -2.9 as shown in warmstarting optimization paper
# gridSearch(objectiveFunction, graph, RawBestCuts[0,0], 3, gamma_step_size=0.1, gammaStart=-3.4, gammaEnd=-2.89, beta_step_size=0.1,betaStart=-3.4, betaEnd=-2.89, plot=True )