import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.interpolate as si
import numpy.linalg as alg
from math import isfinite
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# This code reproduces the algorithm describe in arXiv:1205.3996v4
# It describes a scalar field under small chemical potential
# The vector for the position must be given as a list of  dimension dim


############################ Parameters #######################################
d = 2           # Dimension of the spatial lattice - d >= 2
mass = 1           # Mass of the complex scalar field
mu = 0.1            # Chemical potential
interaction = 1    # Value of lambda for the interaction part 
L = 3             # Lattice size x is in [0,L-1]^dim
#epsilon = 1e-4     # Stepsize for the noise - suff. small for hess. approx
T = 4             # Final time for the integration
n = int(L**d)
dr0 = 0.05        # Stepsize for integration
dr_max=0.05
precision = 1e-7
f = 0.95
N=10

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
    phi1_list = []
    phi2_list = []
    phi1,phi2 = phi1_0,phi2_0
    while r<T and finite:
        # we make the calculation at dr and dr/2 and see if the diff d
        # between the two obtained value is below the precision
        phi1_next,phi2_next = RK4_step(phi1,phi2,r,dr)
        phi1_tmp,phi2_tmp = RK4_step(phi1,phi2,r,dr*0.5)
        phi1_next_half,phi2_next_half = RK4_step(phi1_tmp, phi2_tmp,
                                                 r+dr*0.5,dr*0.5)
        tmp1 = alg.norm(phi1_next-phi1_next_half)
        tmp2 = alg.norm(phi2_next-phi2_next_half)
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
                r = T
                r_list.append(r)
                phi1_list.append(phi1)
                phi2_list.append(phi2)
            else:
                phi1,phi2 = phi1_next, phi2_next
                r_list.append(r)
                phi1_list.append(phi1)
                phi2_list.append(phi2)
        else:
            dr = f*dr*(precision/tmp)**0.2
    return phi1, phi2, r_list, phi1_list, phi2_list, finite

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
    return S(phi1,phi2) - np.log(abs(det_jac))

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

def jacobian_evolution(jac0,r_list,phi1_list,phi2_list):
    jac = jac0
    for i in range(len(r_list)-1):
        dr = r_list[i+1]-r_list[i]
        jac += np.conj(np.matmul(Hessian(phi1_list[i],phi2_list[i]),jac))*dr
    return jac
########################################################################

def Moyenne(ensemble):
    result = 0
    N = len(ensemble)
    for i in range(N):
        result+=ensemble[i]
    return result/N

def Jackknife(ensemble):
    final = []
    N = len(ensemble)
    for i in range(len(ensemble)):
        result = 0
        for j in range(len(ensemble)):
            if j!=i:
                result += ensemble[j]
        result = result/(N-1)
        final.append(result)
    return final

def Variance(ensemble):
    result = 0
    N = len(ensemble)
    moy = Moyenne(ensemble)
    for el in ensemble:
        result += (el - moy)**2
    return result*(N-1)/N

def N_part_estimator(phi1,phi2):
    N = 0
    phi1_neig = neighbour_pos(phi1,0)
    phi2_neig = neighbour_pos(phi2,0)
    for i in range(n):
        N+= np.sinh(mu)*phi1[i]*phi1_neig[i]\
        - 1j*(phi1[i]*phi2_neig[i]-phi2[i]*phi1_neig[i])*np.cosh(mu)
    return N

def plot_npart(mu,T,skip):
    with open("./data/config_mu_{0}_T_{1}.txt".format(mu,T),"rb") as fp:
        config=pickle.load(fp)
    del config[0]

    phi1 = []
    phi2 = []
    det = []

    for el in config:
        phi1.append(el[0])
        phi2.append(el[1])
        det.append(el[2])

    Z = 0
    # Number of particle
    N_part = []
    for i in range(len(config)):
        N_part.append(N_part_estimator(phi1[i],phi2[i])*np.exp(1j*np.angle(det[i])-np.imag(S(phi1[i],phi2[i]))))
        Z += np.exp(1j*np.angle(det[i])-np.imag(S(phi1[i],phi2[i])))
    N_part = np.array(N_part)/Z

    #### Here I have to jackknife or Gamma
    N_skip = []
    for i in range(len(N_part)):
        if i%skip == 0:
            N_skip.append(N_part[i])

    jack_list = Jackknife(N_skip)

    meanN = Moyenne(jack_list)
    varN = Variance(jack_list)
    sigma = np.sqrt(varN)

    print("Nb particles: {0} +/- {1}".format(meanN,sigma))
    return meanN, sigma, N_part


##### Main part #####

class Labeloffset():
    def __init__(self,  ax, label="", axis="y"):
        self.axis = {"y":ax.yaxis, "x":ax.xaxis}[axis]
        self.label=label
        ax.callbacks.connect(axis+'lim_changed', self.update)
        ax.figure.canvas.draw()
        self.update(None)

    def update(self, lim):
        fmt = self.axis.get_major_formatter()
        self.axis.offsetText.set_visible(False)
        self.axis.set_label_text(self.label + " "+ fmt.get_offset() )

#### Plot of the mean number of particles at given mu
mu_list = np.array([i for i in range(1,10)])
mu_list = mu_list/10
ImN = []
Ims = []
T = 0.6
fig1, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)
ax2.set_xlabel(r"Value of the chemical potential $\mu$ wrt the mass: $\frac{\mu}{m}$")
ax1.set_ylabel(r"Particle nb. expect. value $\langle n \rangle$")
ax2.set_ylabel(r"Absolute error on $\langle n \rangle$")
ax2.set_yscale("log")
props = dict(boxstyle='round',facecolor='whitesmoke',alpha=0.7)
text = "Integration time T={0}\n".format(T) + r"$d=2$ - $L=3$"
fig1.text(0.55,0.55, text ,transform=ax2.transAxes,fontsize=13,verticalalignment='bottom',bbox=props)
ax1.grid()
ax2.grid()
fig1.subplots_adjust(hspace=.0)
color_list = ['red','green','blue','grey','orange','gray','greenforest']
skip_list = [1,10,20,40,100,200]
for i in range(len(skip_list)):
    skip = skip_list[len(skip_list)-1-i]
    color = color_list[i]
    meanN = []
    sigmaN = []
    for mu in mu_list:
        m_tmp, s_tmp, N_part = plot_npart(mu,T,skip)
        meanN.append(np.real(m_tmp))
        sigmaN.append(np.real(s_tmp))

    meanN = np.array(meanN)
    sigmaN = np.array(sigmaN)

    ax1.errorbar(mu_list,meanN,yerr=sigmaN,color=color,elinewidth=2,fmt='o',alpha=0.6,solid_capstyle='projecting',capsize=10,label=r"$N_{skip} =$"+"{0}".format(skip))
    shift = 0.025
    for i in range(len(mu_list)):
        mu_tmp = [mu_list[i]-shift,mu_list[i]+shift]
        mean_N_top = [meanN[i]+sigmaN[i],meanN[i]+sigmaN[i]]
        mean_N_bottom = [meanN[i]-sigmaN[i],meanN[i]-sigmaN[i]]
        sigma_tmp = [sigmaN[i],sigmaN[i]]
        axis0 = [0, 0]
        ax1.fill_between(mu_tmp,mean_N_top,mean_N_bottom,color=color,alpha=0.2)
        ax2.fill_between(mu_tmp,sigma_tmp, axis0 ,color=color,alpha=0.6)
ax1.legend(loc='best')

lo = Labeloffset(ax2,label=r"Absolute error on $\langle n \rangle$",axis='y')

fig1.savefig("meanN_mu.pdf",format='pdf',bbox_inches='tight')

##### Plot the imaginary part of the action ######

def plot_imS(mu,T):
    with open("./data/config_mu_{0}_T_{1}.txt".format(mu,T),"rb") as fp:
        config=pickle.load(fp)
    del config[0]

    phi1 = []
    phi2 = []
    det = []

    for el in config:
        phi1.append(el[0])
        phi2.append(el[1])
        det.append(el[2])

    ImS_list = []
    for i in range(len(config)):
        ImS_list.append(np.imag(S(phi1[i],phi2[i])))
    ImS_list = np.array(ImS_list)

    return ImS_list.mean(),ImS_list.std()

mu = 0.9
fig1, ax1 = plt.subplots(nrows=1,ncols=1)
ax1.set_xlabel("Integration time T")
ax1.set_ylabel(r"Imaginary part of the action S")
props = dict(boxstyle='round',facecolor='whitesmoke',alpha=0.7)
fig1.text(0.3,0.8, r"Run at chemical potential $\mu=${0}".format(mu)+"\n$d=2$ - $L=3$",transform=ax2.transAxes,fontsize=13,verticalalignment='bottom',bbox=props)
ax1.grid()
color = 'blue'
T_list = np.array(range(2,62,4))/100
mS,stdS = [], []
for i in range(len(T_list)):
    T = T_list[i]
    meanS, sS = plot_imS(mu,T)
    mS.append(meanS)
    stdS.append(sS)

ax1.errorbar(T_list,mS,yerr=stdS,color=color,elinewidth=2,fmt='o',alpha=0.6,solid_capstyle='projecting',capsize=10)
shift = 0.025
for i in range(len(T_list)):
    T_tmp = [T_list[i]-shift,T_list[i]+shift]
    mean_N_top = [mS[i]+stdS[i],mS[i]+stdS[i]]
    mean_N_bottom = [mS[i]-stdS[i],mS[i]-stdS[i]]
    sigma_tmp = [stdS[i],stdS[i]]
    ax1.fill_between(T_tmp,mean_N_top,mean_N_bottom,color=color,alpha=0.2)

fig1.savefig("meanS_T.pdf",format='pdf',bbox_inches='tight')
