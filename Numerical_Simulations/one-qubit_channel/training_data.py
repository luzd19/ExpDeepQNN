from qutip import*
import numpy as np
from pylab import*
from forward_backward import forward_th

par_target = array([ [3.32176, 6.10338, -1.05099, 6.22025, 2.57295], [-1.45781, -6.23323, 0.344142, -5.4265, 5.55959] ]) ;

# 训练集
train_psi_1 = ket2dm( basis(2, 0) );

train_psi_2 = ket2dm( basis(2, 1) );

train_psi_3 = ket2dm( (basis(2, 0) - basis(2, 1)).unit() );

train_psi = [train_psi_1, train_psi_2, train_psi_3] ;  size_train = len(train_psi) ; 

train_psi_out = [] ;

for p in range(size_train): 
    phiout_r, _ = forward_th(par_target, train_psi[p])  ;
    train_psi_out.append( phiout_r[-1] ) ;
