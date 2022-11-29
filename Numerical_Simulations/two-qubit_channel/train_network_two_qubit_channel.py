import numpy as np
from numpy import pi,linspace
import cmath, random
from pylab import *
from training_data import *
from forward_backward import *

eps = 0.15 ;  iters = 60;

def mean_fidelity(X):
    total_fidelity = 0 ;
    for i in range(size_train): 
        phiout, _ = forward_th(X, train_psi[i]) ;    
        total_fidelity = total_fidelity + fidelity(phiout[-1], train_psi_out[i])
    return total_fidelity/size_train



def train_network(pram_x, eps, iters):
    
    learning_curve = [] ;
    pram_x_history = [] ;  pram_x_history.append(pram_x) ;
    
    for j in range(iters):
        
        X = pram_x_history[-1] ; # X 是门参数 
        learning_curve.append(mean_fidelity(X)) ;
        
        gra_x = np.zeros((num_varia_perceptron, num_perceptron_total));
        for p in range(size_train):
            
            # forward
            phiout, block = forward_th(X, train_psi[p]) ;    #  phiout中的每一个元素依次代表qnn的每一层前向输出
            
            # backward
            H = train_psi_out[p].sqrtm() * ( (train_psi_out[p].sqrtm() * phiout[-1] * train_psi_out[p].sqrtm()).sqrtm() ).inv() * train_psi_out[p].sqrtm()          
            sigma = backward(X, H)

            # gradient
            for i in range(num_layer):
                
                for t in range(num_perceptron_layer):

                    for k in range(num_varia_perceptron):
                        X_s = X+0; X_s[k, 4*i+t] = X[k, 4*i+t] + pi ;
                        partial_U = U( X_s[:,4*i:4*i+4] ) ;
                        U_dag = block[i].dag() ;
                        rho = tensor( phiout[i], ket2dm( tensor(basis(2, 0), basis(2, 0)) ) ) ;
                        back_term = tensor( tensor(qeye(2), qeye(2)), sigma[i+1] ) ;
                        
                        gra_x[k,4*i+t] = gra_x[k,4*i+t] + real( (partial_U * rho * U_dag * back_term ).tr()  ) ; 


        gra_x = gra_x/size_train    
            
        pram_x_history.append( mod( X + eps * gra_x, 4*pi) ) ; 
        
    return pram_x_history, learning_curve
