from qutip import*
import numpy as np
from pylab import*
from forward_backward import forward_th

par_target = array([ [4.75128, -1.46248, -3.61362, 4.56612, -2.05449, 6.0257, -3.46955, -1.7135], [-1.90875, 6.13849, -0.967065, 1.607, -0.397511, -2.32345,  -0.309164, 0.921505] ]) ;

# 训练集
train_psi_1 = ket2dm( tensor(basis(2, 0), basis(2, 0)) );

train_psi_2 = ket2dm( tensor(basis(2, 0), basis(2, 1)) );

train_psi_3 = ket2dm( tensor((basis(2, 0) + basis(2, 1)).unit(), (basis(2, 0) + basis(2, 1)).unit()) );

train_psi_4 = ket2dm( tensor((basis(2, 0) + 1j * basis(2, 1)).unit(), (basis(2, 0) + 1j * basis(2, 1)).unit()) );

train_psi = [train_psi_1, train_psi_2, train_psi_3, train_psi_4] ;  size_train = len(train_psi) ; 

train_psi_out = [] ;

for p in range(size_train): 
    phiout_r, _ = forward_th(par_target, train_psi[p])  ;
    train_psi_out.append( phiout_r[-1] ) ;


    
    
