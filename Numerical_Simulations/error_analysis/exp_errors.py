import numpy as np
from qutip import *
import cmath, copy
from pylab import *
from copy import deepcopy
from scipy import optimize,integrate

M=2; M2=2

### defined parameters
B1 = tensor(qeye(M), qeye(M), qeye(M), qeye(M), qeye(M), qeye(M), destroy(M2), qeye(M2))
B2 = tensor(qeye(M), qeye(M), qeye(M), qeye(M), qeye(M), qeye(M), qeye(M2), destroy(M2))
q1 = tensor(destroy(M), qeye(M), qeye(M), qeye(M), qeye(M), qeye(M), qeye(M2), qeye(M2))
q2 = tensor(qeye(M), destroy(M), qeye(M), qeye(M), qeye(M), qeye(M), qeye(M2), qeye(M2))
q3 = tensor(qeye(M), qeye(M), destroy(M), qeye(M), qeye(M), qeye(M), qeye(M2), qeye(M2))
q4 = tensor(qeye(M), qeye(M), qeye(M), destroy(M), qeye(M), qeye(M), qeye(M2), qeye(M2))
q5 = tensor(qeye(M), qeye(M), qeye(M), qeye(M), destroy(M), qeye(M), qeye(M2), qeye(M2))
q6 = tensor(qeye(M), qeye(M), qeye(M), qeye(M), qeye(M), destroy(M), qeye(M2), qeye(M2))

qlist = [q1, q2, q3, q4, q5, q6] ;

qbus1 = [1, 3, 2, 4] ; qbus2 = [3, 5, 4, 6] ; qbuslist = [qbus1, qbus2] ;


## construct the master equation for quantum state evolution

## qubit relaxation
n_th_list = array([0, 0, 0, 0, 0, 0]) ;

def decohere(times):
    c_ops = [] ; 
    T1_list = 10**(times) * array([8, 6, 5, 8, 10, 10])  ; T2_list = 10**(times) * array([1,1,6,8,12,9]) ; # us
    for qid in range(6):
        gamma = 1/1e3/T1_list[qid]
        gammap = 1/1e3/T2_list[qid] - gamma/2
        n_th = n_th_list[qid]   ####
        q = qlist[qid]
        rate_b1 = gamma * (n_th + 1)   ####
        c_ops.append(sqrt(rate_b1)*q)
        rate_b2 = gamma * n_th
        c_ops.append(sqrt(rate_b2)*q.dag())
        rate_b3 = gammap * 2
        c_ops.append(sqrt(rate_b3)*q.dag()*q)
    return c_ops
    
c_ops = decohere(0) ;

    
# residual ZZ interactions between qubits.
list = [ [[1,3], [2,3], [1,4], [2,4]],   [[3,5], [4,5], [3,6], [4,6]] ] ;
depth = 3 ;
num_layer = depth - 1 ;
num_perceptron_layer = len(list[0]) ;

def H_interact(a, b, f):
    return f * a.dag() * a * b.dag() * b 

def generate_H0(f):
    H0 = 0 ;
    H0 = tensor(qeye(M),qeye(M),qeye(M),qeye(M),qeye(M),qeye(M),qeye(M2),qeye(M2)) ;
    for i in range(num_layer):
        qbus = qbuslist[i] ;
        for j in range(0,num_perceptron_layer-1):
            for k in range(j+1,num_perceptron_layer):
                H0 = H0 + H_interact( qlist[qbus[j]-1], qlist[qbus[k]-1], f[qbus[j],qbus[k]]) 
    return H0



mysigma = 10

def gauss0(x):
	return exp(-(x / 1) ** 2/2)

def coeffq1(t, args):
    offset = 0
    mysigma = args['mysigma']
    detu = args['Freq_d'] - args['Freq_rot']
    t0 = args['tot_time'] 
    pfactor = 1
    A1, lim = integrate.quad(gauss0, -2, 2)
    return  pfactor/A1*gauss0(t/mysigma-offset-2)/mysigma*exp(-1j*(t+t0)*detu)

def coeffq2(t, args):
    offset = 0
    mysigma = args['mysigma']
    detu = args['Freq_d'] - args['Freq_rot']
    t0 = args['tot_time']
    pfactor = 1
    A1, lim = integrate.quad(gauss0, -2, 2)
    return  pfactor/A1*gauss0(t/mysigma-offset-2)/mysigma*exp(1j*(t+t0)*detu)

def qrotate(qdown, axisPhi, angle, H0):
    H_t = deepcopy(H0)
    sigmaPlus = qdown.dag()/2
    temp_sp = sigmaPlus * angle * exp(1j * axisPhi)
    H_t.append([temp_sp, coeffq1])
    H_t.append([temp_sp.dag(), coeffq2])
    return H_t
