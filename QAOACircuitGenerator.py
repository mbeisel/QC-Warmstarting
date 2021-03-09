from qiskit.circuit.quantumcircuit import QuantumCircuit

class QAOACircuitGenerator():
    @classmethod
    def genQAOAcircuit(cls, params, G, p):
        # prepare the quantum and classical resisters
        V = G.nodes()
        E = G.edges()
        QAOA = QuantumCircuit(len(V), len(V))
        
        # apply the layer of Hadamard gates to all qubits
        QAOA.h(range(len(V)))
        QAOA.barrier()
        
        for iter in range(p):
            # apply the Ising type gates with angle gamma along the edges in E
            for edge in E:
                k = edge[0]
                l = edge[1]
                w = G[k][l]['weight']
                QAOA.cp(-2*params[2*iter]*w, k, l)
                QAOA.p(params[2*iter], k)
                QAOA.p(params[2*iter], l)
                
            # then apply the single qubit X rotations with angle beta to all qubits
            QAOA.barrier()
            QAOA.rx(2*params[2*iter+1], range(len(V)))
        
        # Finally measure the result in the computational basis
        QAOA.barrier()
        QAOA.measure(range(len(V)),range(len(V)))
        
        ### draw the circuit for comparison
        #QAOA.draw()
        return QAOA