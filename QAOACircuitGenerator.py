from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt
from builtins import isinstance


class QAOACircuitGenerator():
    @classmethod
    def genQaoaMaxcutCircuitTemplate(cls, graph, initial = None, p=1):
        # prepare the quantum and classical resisters
        n_vertices = graph.shape[0]
        QAOA = QuantumCircuit(n_vertices, n_vertices)


        # prepare the parameters
        gammas = [Parameter("gamma"+str(i+1)) for i in range(p)]
        betas = [Parameter("beta"+str(i+1)) for i in range(p)]

        if(initial):
            for qubits in range(n_vertices):
                QAOA.ry(2*np.arcsin(np.sqrt(initial[qubits])), qubits)
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
                        #pennylane
                        QAOA.cx(i, j)
                        QAOA.rz(-gammas[iter]*graph[i,j], j)
                        QAOA.cx(i, j)
                        # controlled z instead of phase
                        # QAOA.crz(-gammas[iter]*graph[i,j], i, j)
                        # QAOA.rz(gammas[iter], i)
                        # QAOA.rz(gammas[iter], j)
                        #google
                        # QAOA.rzz( -2* gammas[iter]*graph[i,j],i, j)
                        #testattempt
                        # QAOA.rz(gammas[iter], i)
                        # QAOA.rz(gammas[iter], j)
                        # original attempt
                        # QAOA.cp(-2*gammas[iter]*(graph[i,j] ), i, j)
                        # QAOA.p(gammas[iter]*1*(graph[i,j] ), i)
                        # QAOA.p(gammas[iter]*1*(graph[i,j] ), j)

            # then apply the single qubit X rotations with angle beta to all qubits
            QAOA.barrier()

            if(initial):
                for qubits in range(n_vertices):
                    QAOA.ry(2*np.arcsin(np.sqrt(initial[qubits])),qubits)
                    QAOA.rz(-2*betas[iter],qubits)
                    QAOA.ry(-2*np.arcsin(np.sqrt(initial[qubits])),qubits)
            else:
                QAOA.rx(2*betas[iter], range(n_vertices))
        
        # Finally measure the result in the computational basis
        QAOA.barrier()
        QAOA.measure(range(n_vertices),range(n_vertices))
        
        ## draw the circuit for comparison
        # QAOA.draw(output='mpl')
        # plt.show()

        return QAOA

    @classmethod
    def genQaoaMaxcutCircuit(cls, graph, params, initial = None, p=1):
        template = QAOACircuitGenerator.genQaoaMaxcutCircuitTemplate(graph, initial[::-1] if initial else None, p)
        return QAOACircuitGenerator.assignParameters(template, params)

    @classmethod
    def assignParameters(cls, circuit_template, params):
        if not isinstance(params, dict):
            parameter_dict = {}
            parameters = circuit_template.parameters
            for i in range(len(params)):
                    if i % 2 == 0:
                        # gamma
                        param_name = "gamma"+str((i//2)+1)
                        parameter = ([x for x in parameters if x.name == param_name])[0]
                        parameter_dict[parameter] = params[i]
                    else:
                        # beta
                        param_name = "beta"+str((i//2)+1)
                        parameter = ([x for x in parameters if x.name == param_name])[0]
                        parameter_dict[parameter] = params[i]
            params = parameter_dict
        return circuit_template.assign_parameters(params)
