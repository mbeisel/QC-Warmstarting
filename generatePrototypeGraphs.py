import numpy as np
from maxcutQaoa import cost_function_C
from graphGenerator import GraphGenerator, GraphPlotter
from graphStorage import GraphStorage
from goemansWilliamson import bestGWcuts

gen_cuts = 150 # generate this many cuts
store_cuts = 30 # store only this many of the best distinct cuts

for nodes in [12,24]:
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