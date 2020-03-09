import numpy as np
import pickle
import scipy.special as sp
from scipy.optimize import fmin

expect_full = []
expect_with_full = []

#for i in range(1):
#    with open("data/expectation_{0}_False.txt".format(i),"rb") as fp:
#        expect_full.append(pickle.load(fp))
#    with open("data/expectation_{0}_True.txt".format(i),"rb") as fp:
#        expect_with_full.append(pickle.load(fp))
expect_theo = 1j*sp.j1(1)/sp.j0(1)
#R = len(expect_full)
#Nr = len(expect_full[0])
#N = Nr*R

with open("data/config_mu_0.1_T_4.0.txt","rb") as fp:
    expect_full = pickle.load(fp)

#with open("data/expectation_Nt20_beta1_step10_True.txt","rb") as fp:
#    expect_full = pickle.load(fp)

N = len(expect_full)
print('N=',N)
R = 0
Nr = N

def Moyenne(list_obs):
    result = 0
    #for sublist in list_obs:
    for el in list_obs:
        result += el
    return result/N

def Gamma(list_obs,t):
    """ t is in [0,Nr]"""
    result = 0
    moyenne = Moyenne(list_obs)
    for i in range(0,Nr-t):
#        for r in range(0,R):
        result += (list_obs[i]-moyenne)*(list_obs[i+t]-moyenne)
    return result/(N-R*t)

def CF(list_obs,W):
    sumoverW = 0
    for t in range(0,W):
        sumoverW = Gamma(list_obs,t)
    return Gamma(list_obs,0) + 2*sumoverW

def tau(list_obs,W):
    return CF(list_obs,W)/2/Gamma(list_obs,0)

def ErrorFunction(list_obs,W):
    return (np.exp(-W/tau(list_obs,W))+ 2*np.sqrt(W/N))/2

test = 10
for W in range(20):
    tmp = ErrorFunction(expect_full,W)
    print(tmp)
    if tmp<test:
        test = tmp
        W0 = W
print(W0)

print('Correlation time is tau=',CF(expect_full,W0)/2/Gamma(expect_full,0))
print('Result: exp(i\phi)={0}+/-{1}'.format(Moyenne(expect_full),np.sqrt(CF(expect_full,W0)/N)))
print('theoretical value :',expect_theo)
