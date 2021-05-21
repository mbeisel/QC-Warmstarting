import networkx as nx  # tool to handle general Graphs 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from scipy.sparse import csr_matrix

class GraphGenerator():
    @classmethod
    def genButterflyGraph(cls):
        # Generating the butterfly graph with 5 nodes 
        n = 5
        V = np.arange(0,n,1)
        E =[(0,1,1.0),(0,2,1.0),(1,2,1.0),(3,2,1.0),(3,4,1.0),(4,2,1.0)] 
        
        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_weighted_edges_from(E)
        
        return nx.adjacency_matrix(G)

    @classmethod
    def genDiamondGraph(cls):
        # Generating the diamond graph with 4 nodes
        n = 4
        V = np.arange(0,n,1)
        E =[(0,1,1.0),(0,2,1.0),(0,3,1.0),(1,2,1.0),(2,3,1.0)]

        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_weighted_edges_from(E)

        return nx.adjacency_matrix(G)

    @classmethod
    def genGridGraph(cls, height, width):
        # Generating the grid graph

        n = height*width
        V = np.arange(0, n, 1)
        E = []
        for node in range(V.size):
            #Check if node has a node above:
            if(node % height != height-1):
                E.append((node,node + 1, 1.0))
                # check if graph has node to its right
            if((n-1) - node >= height ):
                E.append((node,node +  height, 1.0))

        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_weighted_edges_from(E)
        
        return nx.adjacency_matrix(G)

    @classmethod
    def genMustyGraph(cls):
        n = 5
        V = np.arange(0, n, 1)
        E = [(0,1,1.0),(0,2,5.0),(1,2,7.0),(1,3,2.0),(2,3,4.0),(3,4,3.0)]
    
        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_weighted_edges_from(E)
    
        return nx.adjacency_matrix(G)

    @classmethod
    def genFullyConnectedGraph(cls, n_vertices, weightRange=(-10, 10)):
        V = np.arange(0, n_vertices, 1)
        E = []
        for node in range(n_vertices):
            for connection in range (node+1, n_vertices):
                E.append((node, connection, np.random.randint(weightRange[1]+1 - weightRange[0]) + weightRange[0]))

        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_weighted_edges_from(E)

        return nx.adjacency_matrix(G)

    @classmethod
    def genRandomGraph(cls, n_vertices, n_edges, weightRange=(-10, 10)):
        matrix = np.zeros((n_vertices, n_vertices))
        graph = None
        while (not graph or not nx.is_connected(graph)):
            edges = np.zeros((n_vertices*(n_vertices-1))//2)
            indices = np.random.choice(range(len(edges)), n_edges, replace=False)
            edges[indices] = 1  #add weights here
            weights = list(np.random.choice(range(weightRange[0], weightRange[1]+1), len(edges)))

            for i in range(1, n_vertices):
                for j in range(n_vertices -1):
                    if i > j:
                        weight, weights = pop(weights)
                        matrix[i,j], edges = pop(edges)
                        matrix[i,j] *= weight
                        matrix[j,i] = matrix[i,j]
            graph = nx.from_numpy_matrix(np.array(matrix))
        return matrix


    @classmethod
    def genWarmstartPaperGraph(cls):
        matrix = np.zeros((6, 6))
        orderedEdgeWeightlist = [3,3,6,9,1,   4,4,-8,4,        3,-7,1,   -7,6,   -5]
        for i in range( matrix.shape[0]-1):
            for j in range(1,matrix.shape[0]):
                if i < j:
                    weight, orderedEdgeWeightlist = pop(orderedEdgeWeightlist)
                    matrix[i,j] = weight
                    matrix[j,i] = matrix[i,j]
        return matrix

    @classmethod
    def genRegularGraph(cls, n_vertices, degree, weightRange=(-10, 10)):
        graph = None
        while (not graph or not nx.is_connected(graph)):
            graph = nx.generators.random_graphs.random_regular_graph(degree, n_vertices)

        for (u,v,w) in graph.edges(data=True):
            w['weight'] = np.random.choice(range(weightRange[0], weightRange[1]+1))
        return nx.adjacency_matrix(graph)

def pop(list):
    firstElement = list[0]
    list = list[1:]
    return (firstElement, list)

class GraphPlotter():
    @classmethod
    def plotGraph(cls, G, printWeights=True, x=None, fname=None):
        if isinstance(G, csr_matrix) or isinstance(G, np.ndarray):
            G = nx.Graph(G)
        if not x:
            colors = ['silver' for _ in G.nodes()]
        else:
            colors = ['r' if int(cls) == 0 else 'b' for cls in x]
        default_axes = plt.axes(frameon=True)
        pos          = nx.circular_layout(G)

        edgeColors = [w.get('weight') for (u,v,w) in G.edges(data=True)]
        nodes = nx.draw_networkx_nodes(G,pos,node_color=colors, node_size=600, alpha=1, ax=default_axes)
        edges = nx.draw_networkx_edges(G,pos,edge_color=edgeColors, edge_cmap=cm.get_cmap("coolwarm"),edge_vmin=-10, edge_vmax=10)

        label_dict= { i : list(range(G.number_of_nodes()))[i] for i in range(0, len(list(range(G.number_of_nodes()))) ) }
        labels = nx.draw_networkx_labels(G, pos, labels={n:lab for n,lab in label_dict.items() if n in pos})
        plt.colorbar(edges)
        if printWeights:
            labels = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

        if fname:
            plt.savefig(fname, format="png")
            plt.close()
        else:
            plt.show()

g = GraphGenerator.genRandomGraph(12,11)
# g = GraphGenerator.genRegularGraph(16, 3)
GraphPlotter.plotGraph(g, printWeights=False)