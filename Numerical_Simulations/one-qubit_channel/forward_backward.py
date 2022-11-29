from qutip import*
import numpy as np

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import phasegate, rx, gate_sequence_product, cnot, sqrtswap, csign
from scipy import optimize,integrate

list = [ [[1,3]], [[3,2]], [[2,4]], [[4,6]], [[6,5]] ] ;


depth = 6 ;
width = 1 ;
num_qubit_total = depth * width ;
num_layer = depth - 1 ;
num_qubit_layer = width * 2 ;   # 2
num_perceptron_layer = len(list[0]) ;  # 1
num_perceptron_total = num_perceptron_layer * num_layer ;
num_varia_perceptron = 2 ;



def U(para_x):
    
    qc = QubitCircuit( N = num_qubit_layer ) ;
    for i in range(num_perceptron_layer):
        qc.add_gate("RX", targets = [0], arg_value = para_x[0][i] ) ;
        qc.add_gate("RX", targets = [1], arg_value = para_x[1][i] ) ;
        qc.add_gate("CSIGN", controls = 0, targets = 1 ) ;
           
    result = gate_sequence_product( qc.propagators() ) ;
    return result


def forward_th(X, phi_in):            # 理论
    phiout_r = [] ;    block = [] ; 
    phiout_r.append( phi_in )  ;

    for j in range(num_layer):  
        
        out = tensor( phiout_r[j], ket2dm( basis(2, 0) ) )  ;
        result = U( X[:, num_perceptron_layer*j : num_perceptron_layer*(j+1)] ) ;
        
        block.append( result ) ;
        out = (result * out * result.dag()).ptrace([2-1])
        phiout_r.append(out)       

    return phiout_r, block 


K = tensor( qeye(2), ket2dm( basis(2, 0) ) )  ;

def backward(X, sigma_out):
    back_r = [] ;  
    back_r.insert(0, sigma_out)  ;
    
    for j in range(num_layer):  
        
        out = tensor( qeye(2), back_r[0] )  ;
        result = U( X[:,num_perceptron_total - num_perceptron_layer*(j+1) : num_perceptron_total - num_perceptron_layer*j] ) ;

        out = ( K * (result.dag() * out * result) ).ptrace([1-1])  
        back_r.insert(0, out)     

    return back_r



