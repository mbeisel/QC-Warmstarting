import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Gridsearch for p = 1
def gridSearch(objective_fun, Graph, approximation_List, p, step_size=0.2, plot=False, fname=None):
    a_gamma = np.arange(0, np.pi, step_size)
    a_beta = np.arange(0, np.pi, step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)
    a_gamma, a_beta = a_gamma.flatten(), a_beta.flatten()

    F1 = np.array([objective_fun([a_gamma[i], a_beta[i]], Graph, approximation_List, p) for i in range(len(a_gamma))])

    # Grid search for the minimizing variables
    result = np.where(F1 == np.amin(F1))
    gamma, beta = a_gamma[result[0][0]], a_beta[result[0][0]]

    # Plot the expetation value F1
    if plot or fname:
        fig = plt.figure()
        #ax  = fig.gca(projection='3d')

        size = len(np.arange(0, np.pi, step_size))
        a_gamma, a_beta, F1 = a_gamma.reshape(size, size), a_beta.reshape(size, size), F1.reshape(size, size)
        #surf = ax.plot_surface(a_gamma, a_beta, F1, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        #ax.set_zlim(np.amin(F1)-1,np.amax(F1)+1)
        #ax.zaxis.set_major_locator(LinearLocator(5))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax = fig.add_subplot(1,1,1)
        ax.contourf(a_gamma, a_beta, F1, cmap=cm.coolwarm, antialiased=True)
    if (fname):
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

    return np.array([gamma, beta]), np.amin(F1)