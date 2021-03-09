import networkx as nx  # tool to handle general Graphs 
import numpy as np
import matplotlib.pyplot as plt 

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
        
        return G

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
        
        return G

    @classmethod
    def genMustyGraph(cls):
        n = 5
        V = np.arange(0, n, 1)
        E = [(0,1,1.0),(0,2,5.0),(1,2,7.0),(1,3,2.0),(2,3,4.0),(3,4,3.0)]
    
        G = nx.Graph()
        G.add_nodes_from(V)
        G.add_weighted_edges_from(E)
    
        return G

class GraphPlotter():
    @classmethod
    def plotGraph(cls, G, x=None):
        if not x:
            colors = ['r' for node in G.nodes()]
        else:
            colors = ['r' if cls == '0' else 'b' for cls in x]
        default_axes = plt.axes(frameon=True)
        pos          = nx.spring_layout(G)
        nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)
        plt.show()
