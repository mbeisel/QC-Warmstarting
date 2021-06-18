import numpy as np
from maxcutQaoa import cost_function_C
from graphGenerator import GraphGenerator, GraphPlotter
from graphStorage import GraphStorage
from goemansWilliamson import bestGWcuts
from matplotlib import pyplot as plt
from helperFunctions import hammingDistance
import json

gen_cuts = 250 # generate this many cuts
store_cuts = 50 # store only this many of the best distinct cuts
path = "graphs/prototype/multiExperiment"

def genGraph(type="3r", nodes=12, fname="test/graph.txt", gen_cuts=10, store_cuts=5):
    print(f"{i}: {nodes} - {type}")
    if type == "3r":
        graph = GraphGenerator.genRegularGraph(nodes, 3)
    elif type == "rand":
        graph = GraphGenerator.genRandomGraph(nodes, np.sum([i for i in range(nodes)])//2)
    elif type == "fc":
        graph = GraphGenerator.genFullyConnectedGraph(nodes)

    cuts = bestGWcuts(graph, gen_cuts, store_cuts, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
    GraphStorage.store(f"{path}/{type}-{str(nodes)}-graph-{i}.txt", graph)
    GraphStorage.storeGWcuts(f"{path}/{type}-{str(nodes)}-cuts-{i}.txt", cuts)

for i in range(20):
    break # disabled
    """ generate graphs """
    for nodes in [12,24]:
        print(f"{i}: {nodes} - 3r")
        graph = GraphGenerator.genRegularGraph(nodes, 3)
        cuts = bestGWcuts(graph, gen_cuts, store_cuts, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
        GraphStorage.store(f"{path}/3r-{str(nodes)}-graph-{i}.txt", graph)
        GraphStorage.storeGWcuts(f"{path}/3r-{str(nodes)}-cuts-{i}.txt", cuts)
        #GraphPlotter.plotGraph(graph, printWeights=False)
        
        print(f"{i}: {nodes} - rand")
        graph = GraphGenerator.genRandomGraph(nodes, np.sum([i for i in range(nodes)])//2)
        cuts = bestGWcuts(graph, gen_cuts, store_cuts, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
        GraphStorage.store(f"{path}/rand-{str(nodes)}-graph-{i}.txt", graph)
        GraphStorage.storeGWcuts(f"{path}/rand-{str(nodes)}-cuts-{i}.txt", cuts)
        #GraphPlotter.plotGraph(graph, printWeights=False)

        print(f"{i}: {nodes} - fc")
        graph = GraphGenerator.genFullyConnectedGraph(nodes)
        cuts = bestGWcuts(graph, gen_cuts, store_cuts, continuous=False, epsilon=0.0, cost_fun=cost_function_C)  # get raw solutions using epsilon = 0
        GraphStorage.store(f"{path}/fc-{str(nodes)}-graph-{i}.txt", graph)
        GraphStorage.storeGWcuts(f"{path}/fc-{str(nodes)}-cuts-{i}.txt", cuts)
        #GraphPlotter.plotGraph(graph, printWeights=False)

initial_cuts = []
for i in range(20):
    for var in ["3r", "rand", "fc"]:
        for nodes in ["12"]:
            cuts = GraphStorage.loadGWcuts(f"{path}/{var}-{str(nodes)}-cuts-{i}.txt")
            # print(f"{path}/{var}-{str(nodes)}-cuts-{i}.txt")
            maxcut = cuts[-1][1]
            if not maxcut>0:
                print(f"{path}/{var}-{str(nodes)}-cuts-{i}.txt")
                print(f"Division by zero ahead!")
                genGraph(type=var, nodes=int(nodes), fname=f"{path}/{var}-{str(nodes)}-cuts-{i}.txt", gen_cuts=gen_cuts, store_cuts=store_cuts)
                continue
            # print(cuts[-1])
            for j in range(len(cuts)-2, -1, -1):
                ratio = cuts[j][1]/maxcut
                if ratio <= 0.9:
                    initial_cuts.append([
                        f"{path}/{var}-{str(nodes)}-graph-{i}.txt",
                        f"{path}/{var}-{str(nodes)}-cuts-{i}.txt",
                        j,
                        ratio,
                        hammingDistance(cuts[j][0], cuts[-1][0], True)
                        ])
                    # print(f"{cuts[j]} ratio: {ratio}")
                    if ratio < 0.7:
                        print(f"{path}/{var}-{str(nodes)}-cuts-{i}.txt")
                        print(f"{cuts[j]} ratio: {ratio}")
                        genGraph(type=var, nodes=int(nodes), fname=f"{path}/{var}-{str(nodes)}-cuts-{i}.txt", gen_cuts=gen_cuts, store_cuts=store_cuts)
                    break

print(initial_cuts)
fname = f"{path}/initial_cuts-12.txt"
initial_cuts_file = open(fname, "w")
json.dump(initial_cuts, initial_cuts_file)
initial_cuts_file.close()

a_file = open(fname, "r")
output = a_file.read()
print(output)

exit()
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