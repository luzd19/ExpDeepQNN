import numpy as np
from numpy import pi,linspace
import cmath, random
from pylab import *
from forward_backward import *

# hamitonian for H2
g0 = -0.4804 ;
g1 = +0.3435;
g2 = -0.4347;
g3 = +0.5716;
g4 = +0.0910;
g5 = +0.0910;
Hamiltonian = g0 * tensor(qeye(2), qeye(2)) + g1 * tensor(sigmaz(), qeye(2)) + g2 * tensor(qeye(2), sigmaz()) +  \
     g3 * tensor(sigmaz(), sigmaz()) + g4 * tensor(sigmay(), sigmay()) + g5 * tensor(sigmax(), sigmax())   ;

psi_in = ket2dm( tensor(basis(2, 0), basis(2, 0)) ) ;


def expect_energy(X):
    phiout, _ = forward_th(X, psi_in) ;    
    energy_estimate = real((Hamiltonian * phiout[-1]).tr())   ;
    return energy_estimate


def train_network(pram_x, eps, iters):
    
    learning_curve = [] ;
    pram_x_history = [] ;  pram_x_history.append(pram_x) ;
    
    for j in range(iters):
        
        X = pram_x_history[-1] ; 
        learning_curve.append(expect_energy(X)) ;
        
        gra_x = np.zeros((num_varia_perceptron, num_perceptron_total));
  
        # forward
        phiout, block = forward_th(X, psi_in) ;

        # backward
        sigma = backward(X, Hamiltonian) ;

        # gradient
        for i in range(num_layer):

            for t in range(num_perceptron_layer):

                for k in range(num_varia_perceptron):
                    X_s = X + 0; X_s[k, 4*i+t] = X[k, 4*i+t] + pi ;
                    partial_U = U( X_s[:,4*i:4*i+4] ) ;
                    U_dag = block[i].dag() ;
                    rho = tensor( phiout[i], ket2dm( tensor(basis(2, 0), basis(2, 0)) ) ) ;
                    back_term = tensor( tensor(qeye(2), qeye(2)), sigma[i+1] ) ;

                    gra_x[k,4*i+t] = gra_x[k,4*i+t] + real( (partial_U * rho * U_dag * back_term ).tr()  ) ;    
   
        pram_x_history.append( mod( X - eps * gra_x, 4*pi) ) ; 
        
    return pram_x_history, learning_curve



'''
def train_network_2(pram_x, eps, iters):
    
    learning_curve = [] ;
    pram_x_history = [] ;  pram_x_history.append(pram_x) ;
    
    for j in range(iters):
        
        X = pram_x_history[-1] ; # X 是门参数 
        learning_curve.append(expect_energy(X)) ;
        
        gra_x = np.zeros((2,8));
  
        # forward
        phiout, block = forward_th(X, psi_in) ; 

        # backward
        sigma = backward(X, Hamiltonian) ;

        # gradient
        for i in range(2):

            for t in range(4):

                for k in range(2):
                    X_s = X+0; X_s[k, 4*i+t] = X[k, 4*i+t] + pi/2 ; X_s2 = X+0; X_s2[k, 4*i+t] = X[k, 4*i+t] - pi/2
                    partial_U = U( X_s[:,4*i:4*i+4] ) ;  partial_U2 = U( X_s2[:,4*i:4*i+4] ) 
                    
                    rho = tensor( phiout[i], ket2dm( tensor(basis(2, 0), basis(2, 0)) ) ) ;
                    back_term = tensor( tensor(qeye(2), qeye(2)), sigma[i+1] ) ;

                    gra_x[k,4*i+t] = gra_x[k,4*i+t] + real( (partial_U * rho * partial_U.dag() * back_term ).tr()/2 - (partial_U2                                      *rho * partial_U2.dag() * back_term ).tr()/2) ;    
   
        pram_x_history.append( mod( X - eps * gra_x, 4*pi) ) ; 
        
    return pram_x_history, learning_curve
'''



