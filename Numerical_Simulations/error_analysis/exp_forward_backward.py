from qutip import*
import numpy as np
from exp_errors import *
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import phasegate, rx, gate_sequence_product, cnot, sqrtswap, csign
from scipy import optimize,integrate

list = [ [[1,3], [2,3], [1,4], [2,4]], 
        [[3,5], [4,5], [3,6], [4,6]] ] ;

depth = 3 ;
width = 2 ;
num_qubit_total = depth * width ;
num_layer = depth - 1 ;
num_qubit_layer = width * 2 ;
num_perceptron_layer = len(list[0]) ;
num_perceptron_total = num_perceptron_layer * num_layer ;
num_varia_perceptron = 2 ;


def Rx(theta, x, order, H0, c_ops):
    n = order - 1 ; tlist = linspace(0,40,41) ;
    args0 = {}; args0['mysigma'] = 10 ; args0['Freq_rot'] = 6.2*pi*2 ;  args0['tot_time'] = 0 ;  
    
    args0['Freq_d'] = 6.2*pi*2 ;
    
    Ht = qrotate( qlist[n], 0, theta, [H0] ) ; 
    psif = mesolve(Ht, x, tlist, c_ops, args = args0).states[-1]       
    
    return psif 


def CZ(x, order1, order2):
    n1 = order1 - 1 ; n2 = order2 - 1 ;
    qc = QubitCircuit(N = num_qubit_total + 2) ;
    qc.add_gate( "CSIGN", controls = n1, targets = n2 ) ;
    result = gate_sequence_product( qc.propagators() ) ;
    return result * x * result.dag()

def U(para_x):
    
    qc = QubitCircuit( N = num_qubit_layer ) ;
    for i in range(num_perceptron_layer):
        qc.add_gate("RX", targets = [list[0][i][0]-1], arg_value = para_x[0][i] ) ;
        qc.add_gate("RX", targets = [list[0][i][1]-1], arg_value = para_x[1][i] ) ;
        qc.add_gate("CSIGN", controls = list[0][i][0]-1, targets = list[0][i][1]-1 ) ;
           
    result = gate_sequence_product( qc.propagators() ) ;
    return result


def forward_th(X, phi_in):            # theoretical
    phiout_r = [] ;    block = [] ; 
    phiout_r.append( phi_in )  ;

    for j in range(num_layer):  
        
        out = tensor( phiout_r[j], ket2dm( tensor(basis(2, 0), basis(2, 0)) ) )  ;
        result = U( X[:, num_perceptron_layer*j : num_perceptron_layer*(j+1)] ) ;
        
        block.append( result ) ;
        out = (result * out * result.dag()).ptrace([3-1,4-1])
        phiout_r.append(out)       

    return phiout_r, block 

def forward_ex(X, phi_in, H0, c_ops):        # experimental
    phiout_r = [] ;    block = [] ; 
    phiout_r.append( phi_in )  ;
  
    out = tensor( phiout_r[0], ket2dm( tensor(basis(2, 0), basis(2, 0),basis(2, 0), basis(2, 0),basis(2, 0), basis(2, 0)) ) )  ;
    for j in range(num_layer): 
        
        for k in range(num_perceptron_layer):
            
            out = Rx(X[1][num_perceptron_layer*j+k], Rx(X[0][num_perceptron_layer*j+k], out, list[j][k][0], H0, c_ops) , list[j][k][1], H0, c_ops)
            out = CZ(out, list[j][k][0], list[j][k][1])
            
        phiout_r.append( out.ptrace([num_perceptron_layer-2+2*j,num_perceptron_layer-1+2*j]) )
    
    return phiout_r

K = tensor( tensor(qeye(2), qeye(2)), ket2dm( tensor(basis(2, 0), basis(2, 0)) ) ) ;

def backward(X, sigma_out):
    back_r = [] ;  
    back_r.insert(0, sigma_out)  ;
    
    for j in range(num_layer):  
        
        out = tensor( qeye(2), qeye(2), back_r[0] )  ;
        result = U( X[:,num_perceptron_total - num_perceptron_layer*(j+1) : num_perceptron_total - num_perceptron_layer*j] ) ;

        out = ( K * (result.dag() * out * result) ).ptrace([1-1,2-1])  
        back_r.insert(0, out)     

    return back_r

