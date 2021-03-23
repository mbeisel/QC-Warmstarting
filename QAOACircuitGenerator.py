from qiskit.circuit.quantumcircuit import QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt


class QAOACircuitGenerator():
    @classmethod
    def genQAOAcircuit(cls, params, graph, initial = None, p=1):
        # prepare the quantum and classical resisters
        n_vertices = graph.shape[0]
        QAOA = QuantumCircuit(n_vertices, n_vertices)

        if(initial):
            initial = initial[::-1]
            for qubits in range(n_vertices):
                QAOA.ry(2*np.arcsin(np.sqrt(initial[qubits])),qubits)
        else:
            # apply the layer of Hadamard gates to all qubits
            QAOA.h(range(n_vertices))

        for iter in range(p):
            QAOA.barrier()
            # apply the Ising type gates with angle gamma along the edges in E
            for i in range(1, n_vertices):
                for j in range(n_vertices -1):
                    if i > j and graph[i,j] != 0:
                        # print(graph[i,j])
                        QAOA.cp(-2*params[2*iter]*graph[i,j], i, j)
                        QAOA.p(params[2*iter], i)
                        QAOA.p(params[2*iter], j)

            # then apply the single qubit X rotations with angle beta to all qubits
            QAOA.barrier()

            if(initial):
                for qubits in range(n_vertices):
                    QAOA.ry(2*np.arcsin(np.sqrt(initial[qubits])),qubits)
                    QAOA.rz(-2*params[2*iter+1],qubits)
                    QAOA.ry(-2*np.arcsin(np.sqrt(initial[qubits])),qubits)
            else:
                QAOA.rx(2*params[2*iter+1], range(n_vertices))
        
        # Finally measure the result in the computational basis
        QAOA.barrier()
        QAOA.measure(range(n_vertices),range(n_vertices))
        
        ## draw the circuit for comparison
        QAOA.draw(output='mpl')
        plt.show()

        return QAOA