from builtins import isinstance
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from numpy import dtype
import ast

class GraphStorage():
    @classmethod
    def store(cls, filename, matrix):
        if isinstance(matrix, csr_matrix):
            matrix = matrix.toarray()
        elif not isinstance(matrix, np.ndarray):
            import networkx
            if isinstance(matrix, networkx.classes.graph.Graph):
                matrix = networkx.linalg.adjacency_matrix(matrix).toarray()
            else:
                raise Exception("Type not supported: {}".format(type(matrix)))

        np.savetxt(filename, matrix)

    @classmethod
    def load(cls, filename):
        matrix = np.loadtxt(filename)
        return csr_matrix(matrix)

    @classmethod
    def storeGWcuts(cls, filename, gwCuts):
        with open(filename, "w+", encoding="utf-8") as file:
            [file.write(str(cut[0])+", "+str(cut[1])+"\n") for cut in gwCuts]
            file.close()

    @classmethod
    def loadGWcuts(self, filename):
        cuts = []
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                cuts.append(ast.literal_eval(line))
        return cuts
