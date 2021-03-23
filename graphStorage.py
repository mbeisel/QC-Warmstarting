from builtins import isinstance
import numpy as np
import scipy
from scipy.sparse import csr_matrix

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
