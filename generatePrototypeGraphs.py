import numpy as np
from maxcutQaoa import cost_function_C
from graphGenerator import GraphGenerator, GraphPlotter
from graphStorage import GraphStorage
from goemansWilliamson import bestGWcuts
from matplotlib import pyplot as plt

gen_cuts = 150 # generate this many cuts
store_cuts = 30 # store only this many of the best distinct cuts

""" generate graphs """
for nodes in [12,24]:
    break # disabled
    graph = GraphGenerator.genRegularGraph(nodes, 3)
    cuts = bestGWcuts(graph, gen_cuts, store_cuts, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
    GraphStorage.store(f"graphs/prototype/3r-{str(nodes)}-graph.txt", graph)
    GraphStorage.storeGWcuts(f"graphs/prototype/3r-{str(nodes)}-cuts.txt", cuts)
    GraphPlotter.plotGraph(graph, printWeights=False)

    graph = GraphGenerator.genRandomGraph(nodes, np.sum([i for i in range(nodes)])//2)
    cuts = bestGWcuts(graph, gen_cuts, store_cuts, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
    GraphStorage.store(f"graphs/prototype/rand-{str(nodes)}-graph.txt", graph)
    GraphStorage.storeGWcuts(f"graphs/prototype/rand-{str(nodes)}-cuts.txt", cuts)
    GraphPlotter.plotGraph(graph, printWeights=False)

    graph = GraphGenerator.genFullyConnectedGraph(nodes)
    cuts = bestGWcuts(graph, gen_cuts, store_cuts, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
    GraphStorage.store(f"graphs/prototype/fc-{str(nodes)}-graph.txt", graph)
    GraphStorage.storeGWcuts(f"graphs/prototype/fc-{str(nodes)}-cuts.txt", cuts)
    GraphPlotter.plotGraph(graph, printWeights=False)

""" plot graphs """
rows = [12, 24]
columns = ["3r", "rand", "fc"]

fig = plt.figure(1, figsize=(1.5*11.7, 1.4*8.3))
fig.tight_layout()

for row, nodes in enumerate(rows):
    for column, graph in enumerate(columns):
        # print(f"line {row} column {column}")

        ax = fig.add_subplot(len(rows), len(columns), 1 + column + len(columns)*row)
        if row == 0:
            ax.set_title(r"$G_{n,{"+graph+"}}$", fontsize=28)
        if column == 0:
            ax.set_ylabel(f"$n={nodes}$", fontsize=28)

        # ax.set_title(f"row {row} column {column}")
        graph = GraphStorage.load(f"graphs/prototype/{graph}-{str(nodes)}-graph.txt")
        edges = GraphPlotter.plotGraph(graph, printWeights=False, ax=ax)

fig.subplots_adjust(right=0.845)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(edges, cax=cbar_ax, ticks=range(-10,11,5))
cbar.ax.tick_params(labelsize=20)
plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.savefig("graphs/prototype/plotted_graphs.pdf", format="pdf")
plt.show()