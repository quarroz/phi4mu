import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.interpolate as si
import numpy.linalg as alg
from math import isfinite
import sys

# This code reproduces the algorithm describe in arXiv:1205.3996v4
# It describes a scalar field under small chemical potential
# The vector for the position must be given as a list of  dimension dim


############################ Parameters #######################################
d = 2           # Dimension of the spatial lattice - d >= 2
mass = 1           # Mass of the complex scalar field
mu = float(sys.argv[1])          # Chemical potential
interaction = 1    # Value of lambda for the interaction part 
L = 3             # Lattice size x is in [0,L-1]^dim
#epsilon = 1e-4     # Stepsize for the noise - suff. small for hess. approx
T = float(sys.argv[2])             # Final time for the integration
n = int(L**d)
dr0 = 0.01        # Stepsize for integration
dr_max=0.1
precision = 1e-8
f = 0.95
N=100

############################ Useful methods ###################################

# The first step consist in finding the eigenvector in direct space

# Convention in the code - Important to read
# a vector (0,1,2 ... n=L**d=9) represent the point on the lattice in this manner
#   0  1  2   -> axis 0
# ( 3  4  5 ) and a position vector has inverse coordinate[id, ..., i1,i0]
#   6  7  8     so index 2 corresponds to [ 0 2 ] in lattice 3x3
# | axis 1 go to the bottom

def position(m):
    """ Return the position vector on the lattice in function of the place in the vector
    for example the position 6 in basis L=2 and d=3 correspond to
    the position vector [1 1 0]
    It correponds to the conversion from basis 10 to basis L
    """
    global L
    x = np.zeros(d,dtype=int)
    i=0
    while(m>0):
        L,m = int(L),int(m)
        x[d-i-1]=m%L
        m = m//L
        i += 1
    p = np.array(x)
    return p

def index(x):
    """ Return the place in the vector in function of the position
    The reverse operation from position.
    """
    result = 0
    for i in range(len(x)):
        result += x[i]*L**(d-i-1)
    return result

def kron(a,b):
    if a==b:
        return 1
    else:
        return 0

def kronecker_pos(x,nu,y):
    """ Return the kronecker d(x+nu,y). nu cannot be negative.
    x,y are between 0 and n-1, nu between 0 and d-1 """
    xp = position(x)
    yp = position(y)
    nu = int(nu)
    add = np.zeros(d,dtype=np.complex)
    add[d-1-nu] = 1
    result = xp + add
    if (result==yp).all():
        return 1
    else:
        return 0

def kronecker_neg(x,nu,y):
    """ Return the kronecker d(x-nu,y). nu cannot be negative.
    x,y are between 0 and n-1, nu between 0 and d-1 """
    xp = position(x)
    yp = position(y)
    nu = int(nu)
    add = np.zeros(d,dtype=np.complex)
    add[d-1-nu] = -1
    result = xp + add
    if (result==yp).all():
        return 1
    else:
        return 0

def neighbour_pos(lattice,mu):
    """ Return the lattice of the neighbour in positive direction mu with
    increment 1 """
    result = np.zeros(n,dtype=np.complex)
    for i in range(n):
        p = position(i)
        if p[d-1-mu]==L-1:
            p[d-1-mu] = 0
        else:
            p[d-1-mu] += 1
        ind = int(index(p))
        result[i] = lattice[ind]
    return result

def neighbour_neg(lattice,mu):
    """ Return the lattice of the neighbour in negative direction mu with
    increment 1 """
    result = np.zeros(n,dtype=np.complex)
    for i in range(n):
        p = position(i)
        if p[d-1-mu]==0:
            p[d-1-mu] = L-1
        else:
            p[d-1-mu] -= 1
        ind = int(index(p))
        result[i] = lattice[ind]
    return result



def real_evolution(p,q):
    """ Return the value of the "real" part of the derivative of the
    complex conjugated action """
    sumoverspace = 0
    for i in range(n):
        sumoverspace += p[i]**2 + q[i]**2
    a = (2*d+mass**2+interaction*sumoverspace)*p
    b = 0
    for nu in range(d):
        b -= np.cosh(mu*kron(nu,0))*(neighbour_pos(p,nu)+neighbour_neg(p,nu))
        b += 1j*np.sinh(mu*kron(nu,0))*(neighbour_neg(q,nu)+neighbour_pos(q,nu))
    result = a+b
    return result

def imag_evolution(p,q):
    """ Return the value of the "real" part of the derivative of the
    complex conjugated action """
    sumoverspace = 0
    for i in range(n):
        sumoverspace += p[i]**2 + q[i]**2
    a = 2*d + mass**2 + interaction*sumoverspace*q
    b = 0
    for nu in range(d):
        b -= np.cosh(mu*kron(nu,0))*(neighbour_pos(q,nu)+neighbour_neg(q,nu))
        b += 1j*np.sinh(mu*kron(nu,0))*(neighbour_pos(p,nu)-neighbour_neg(p,nu))
    result = a+b
    return result

def delta_phi(phi1,phi2):
    result = 0
    for i in range(n):
        result+= R[i,0,0]*(phi1[i]**2+phi2[i]**2 + 2*phi1[i]**2)
        result+= R[i,1,1]*(phi1[i]**2+phi2[i]**2 + 2*phi2[i]**2)
        result+= R[i,0,1]*(2*phi1[i]*phi2[i])
        result+= R[i,1,0]*(2*phi1[i]*phi2[i])
    return np.conj(result)

def TraceHprim(phi1,phi2):
    return (sum_lambda - delta_phic) + delta_phi(phi1,phi2)

def RK4_step(phi1,phi2,r,dr):
    """ Return the result of one step RK4 starting from pi,qi
    at position r with step dr """
    pi = phi1
    qi = phi2
    p1 = real_evolution(pi,qi)*dr
    q1 = imag_evolution(pi,qi)*dr
    p2 = real_evolution(pi+0.5*p1,qi+0.5*q1)*dr
    q2 = imag_evolution(pi+0.5*p1,qi+0.5*q1)*dr
    p3 = real_evolution(pi+0.5*p2,qi+0.5*q2)*dr
    q3 = imag_evolution(pi+0.5*p2,qi+0.5*q2)*dr
    p4 = real_evolution(pi+p3,qi+q3)*dr
    q4 = imag_evolution(pi+p3,qi+q3)*dr
    p_next = pi + 1.0/6.0*(p1+2.0*p2+2.0*p3+p4) # Evolution of q
    q_next = qi + 1.0/6.0*(q1+2.0*q2+2.0*q3+q4)
    return p_next,q_next

def RK4(phi1_0,phi2_0):
    """ Runge-Kutta order 4 method with adaptative stepsize
    Return the last value p,q, real and imaginary part of the solution
    and the list of r and the corresponding hessian value."""
    finite = True
    r=0
    dr = dr0
    r_list = []
    #phi1_list = []
    #phi2_list = []
    det = sum_lambda
    phi1,phi2 = phi1_0,phi2_0
    while r<T and finite:
        # we make the calculation at dr and dr/2 and see if the diff d
        # between the two obtained value is below the precision
        phi1_next,phi2_next = RK4_step(phi1,phi2,r,dr)
        phi1_tmp,phi2_tmp = RK4_step(phi1,phi2,r,dr*0.5)
        phi1_next_half,phi2_next_half = RK4_step(phi1_tmp, phi2_tmp,
                                                 r+dr*0.5,dr*0.5)
        trH = TraceHprim(phi1_next,phi2_next)*dr
        trH_tmp = TraceHprim(phi1_next_half,phi1_next_half)*dr
        tmp1 = alg.norm(phi1_next-phi1_next_half)
        tmp2 = alg.norm(phi2_next-phi2_next_half)
        tmp3 = abs(trH-trH_tmp)
        tmp = max(tmp1,tmp2)
        if tmp == 0:
            dr = 10*dr
            continue
        if tmp<=precision:
            dr_old = dr
            dr = dr*(precision/tmp)**0.2  # Find the new stepsize
            if dr > dr_max:
                dr = dr_max
            r = r + dr_old
            if r+dr > T:
                dr = T-r
                phi1,phi2 = RK4_step(phi1_next,phi2_next,r,dr)
                det += trH*dr
                r = T
                #r_list.append(r)
                #phi1_list.append(phi1)
                #phi2_list.append(phi2)
            else:
                phi1,phi2 = phi1_next, phi2_next
                det += trH
                #r_list.append(r)
                #phi1_list.append(phi1)
                #phi2_list.append(phi2)
        else:
            dr = f*dr*(precision/tmp)**0.2
    return phi1, phi2, det, finite

def S(phi1,phi2):
    result = 0
    a = (d+mass**2/2+interaction/4)*(phi1**2+phi2**2)
    c = 0
    for nu in range(d):
        c -= np.cosh(mu*kron(nu,0))*(phi1*neighbour_pos(phi1,nu)\
                                     + phi2*neighbour_pos(phi2,nu))
        c-= 1j*np.sinh(mu*kron(nu,0))*(neighbour_pos(phi1,nu)*phi2\
                                       -neighbour_pos(phi2,nu)*phi1)
    for i in range(n):
        result += a[i] + c[i]
    return result

def effective_Action(phi1, phi2 ,det_jac):
    return np.real(S(phi1,phi2)) - np.log(abs(det_jac))

def Metropolism(phi1_old, phi2_old, det_jac_old, phi1_new, phi2_new, det_jac_new):
    test = np.exp(-effective_Action(phi1_new, phi2_new, det_jac_new)\
                  +effective_Action(phi1_old, phi2_old,det_jac_old))
    proba = min(1,test)
    x = np.random.random()
    return x<proba

def Hessian(phi1,phi2):
    """Initialize the hessian matrix for field values phi1,phi2 """
    H = np.zeros((2*n,2*n),dtype=np.complex)
    sumoverallfield = 0
    for i in range(n):
        sumoverallfield += phi1[i]**2 + phi2[i]**2
    for i in range(n):
        for j in range(n):
            H[i,j] += (2*d+mass**2+interaction*sumoverallfield)*kron(i,j)
            H[i,j] -= 2*interaction*phi1[i]*phi1[j]
            sumoverdim = 0
            for nu in range(d):
                sumoverdim+= np.cosh(mu*kron(nu,0))*(kronecker_pos(i,nu,j)\
                            +kronecker_neg(i,nu,j))
            H[i,j] += sumoverdim
        for j in range(n,2*n):
            H[i,j] -= 2*interaction*phi1[i]*phi2[j%n]
            sumoverdim = 0
            for nu in range(d):
                sumoverdim -= 1j*np.sinh(mu*kron(nu,0))*\
                        (kronecker_pos(i,nu,j%n)-kronecker_pos(j%n,nu,i))
            H[i,j] += sumoverdim
    for i in range(n,2*n):
        for j in range(n,2*n):
            H[i,j] += (2*d+mass**2+interaction*sumoverallfield)*kron(i,j)
            H[i,j] -= 2*interaction*phi2[i%n]*phi2[j%n]
            sumoverdim = 0
            for nu in range(d):
                sumoverdim+= np.cosh(mu*kron(nu,0))*(kronecker_pos(i%n,nu,j%n)\
                            +kronecker_neg(i%n,nu,j%n))
            H[i,j] += sumoverdim
        for j in range(n):
            H[i,j] -= 2*interaction*phi1[i%n]*phi2[j]
            sumoverdim = 0
            for nu in range(d):
                sumoverdim -= -1j*np.sinh(mu*kron(nu,0))*\
                        (kronecker_pos(i%n,nu,j)-kronecker_pos(j,nu,i%n))
            H[i,j] += sumoverdim
    return H

def Hessian_critical():
    phi1_c = np.zeros(shape=n,dtype=np.complex)
    phi2_c = np.zeros(shape=n,dtype=np.complex)
    H_c = Hessian(phi1_c,phi2_c)
    Hr = np.real(H_c)
    Hi = np.imag(H_c)
    H_part1 = np.concatenate((Hr,-Hi),axis=0)
    H_part2 = np.concatenate((-Hi,-Hr),axis=0)
    H_todiag = np.concatenate((H_part1,H_part2),axis=1)
    return H_todiag

def proposal(ensemble):
    """ Return a proposal in the positive eigenspace of the hessian """
    proposal = np.random.normal(loc=0.0,scale=1.0,size=n)
    phi_eta = np.zeros(n,dtype=np.complex)
    for i in range(n):
        phi_eta += proposal[i]*ensemble[i]
    return phi_eta

#def jacobian_evolution(jac0,r_list,phi1_list,phi2_list):
#    jac = jac0
#    for i in range(len(r_list)-1):
#        dr = r_list[i+1]-r_list[i]
#        jac += np.conj(np.matmul(Hessian(phi1_list[i],phi2_list[i]),jac))*dr
#    print(jac)
#    return jac
######################################################################

# Determine the positive eigenvalue space of the hessian.
H_c = Hessian_critical()
w,v = alg.eigh(H_c)
positive_eigenspace = []
positive_eigenspace_1 = []
positive_eigenspace_2 = []
sum_lambda = 0
for i in range(len(w)):
    phi1 = np.zeros(n,dtype=complex)
    phi2 = np.zeros(n,dtype=complex)
    if w[i] > 0:
        sum_lambda += w[i]
        positive_eigenspace.append(v[:,i])
        for j in range(n):
            # Note that the eigenvector is v[:,i]
            phi1[j] = v[j,i] + 1j*v[j+2*n,i]
            phi2[j] = v[j+n,i] + 1j*v[j+3*n,i]
        positive_eigenspace_1.append(phi1)
        positive_eigenspace_2.append(phi2)

print('The dimension of the positive eigenspace is'
      ,len(positive_eigenspace_1))

## Construction of Pn
#Pn = np.zeros(shape=(2*n,2*n),dtype=np.complex)
#for i in range(2*n):
#    for j in range(2*n):
#        Pn[j,i] = positive_eigenspace[i][j]

# Here is calculated the stuff needed for the jacobian evo.
# First the Rx_a,b matrix
R= []
for i in range(n):
    Mab = np.zeros((2,2),dtype=np.complex)
    for j in range(len(positive_eigenspace)):
        Mab[0,0] += positive_eigenspace_1[j][i]**2
        Mab[0,1] += positive_eigenspace_1[j][i]\
                *positive_eigenspace_2[j][i]
        Mab[1,0] += Mab[0,1]
        Mab[1,1] += positive_eigenspace_2[j][i]**2
    R.append(Mab)
R = np.array(R)
# Here calculated the delta_phic

phi1_c = np.zeros(n,dtype=np.complex)
phi2_c = np.zeros(n,dtype=np.complex)

delta_phic = delta_phi(phi1_c,phi2_c)

config = []

# Initial config at critical point
phi_0_1 = np.zeros(n,dtype=np.complex)
phi_0_2 = np.zeros(n,dtype=np.complex)
#det_0 = alg.det(Pn)
det_0 = sum_lambda
initial_config = [phi_0_1,phi_0_2,det_0]

config.append(initial_config)
imagS = []

for i in range(N):
    # Here begins the loop for the N iterations
    print('New iteration')
    phi1, phi2 = proposal(positive_eigenspace_1),\
            proposal(positive_eigenspace_2)
    phi1,phi2, det ,finite = RK4(phi1,phi2)
    while not(finite):
        phi1, phi2 = proposal(positive_eigenspace_1),\
                proposal(positive_eigenspace_2)
        phi1, phi2, det, finite =  RK4(phi1,phi2)
    print(det)
    #det = alg.det(jacobian_evolution(Pn,r_list,phi1_list,phi2_list))
    if Metropolism(phi_0_1,phi_0_2,det_0,phi1,phi2,det):
        phi_0_1 = phi1
        phi_0_2 = phi2
        det_0 = det
    config.append([phi_0_1,phi_0_2,det_0])
    imagS.append(np.imag(S(phi_0_1,phi_0_2)))
    print('Done !')
    print('ImS = ', np.imag(S(phi1,phi2)))

with open("data/config_mu_{0}_T_{1}.txt".format(mu,T),"wb") as fp:
    pickle.dump(config,fp)
