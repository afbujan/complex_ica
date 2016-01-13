import pdb,os,time
import numpy as np
from math import log,log10
from numpy.linalg import *
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import complex_FastICA as cica
reload(cica)

plt.ion()


m = 50000
j = 0


# some parameters for the available distributions

bino1 = np.maximum(2, np.ceil(20*rand()))
bino2 = rand()
exp1 = np.ceil(10*rand())
gam1 = np.ceil(10*rand())
gam2 = gam1 + np.ceil(10*rand())
f1 = np.ceil(10*rand())
f2 = np.ceil(100*rand())
poiss1 = np.ceil(10*rand())
nbin1 = np.ceil(10*rand())
nbin2 = rand()
hyge1 = np.ceil(900*rand())
hyge2 = np.ceil(20*rand())
hyge3 = round(hyge1/np.maximum(2,np.ceil(5*rand())))
chi1 = np.ceil(20*rand())
beta1 = np.ceil(10*rand())
beta2 = beta1 + np.ceil(10*rand())
unif1 = np.ceil(2*rand())
unif2 = unif1 + np.ceil(2*rand())
gam3 = np.ceil(20*rand())
gam4 = gam3 + np.ceil(20*rand())
f3 = np.ceil(10*rand())
f4 = np.ceil(50*rand())
exp2 = np.ceil(20*rand())
rayl1 = 10
unid1 = np.ceil(100*rand())
norm1 = np.ceil(10*rand())
norm2 = np.ceil(10*rand())
logn1 = np.ceil(10*rand())
logn2 = np.ceil(10*rand())
geo1 = rand()
weib1 = np.ceil(10*rand())
weib2 = weib1 + np.ceil(10*rand())



r = np.random.binomial(bino1,bino2,size=(1,m))

r = np.vstack([r,np.random.gamma(gam1,gam2,size=(1,m))])
r = np.vstack([r,np.random.poisson(poiss1,size=(1,m))])
r = np.vstack([r,np.random.hypergeometric(hyge1,hyge2,hyge3,size=(1,m))])
r = np.vstack([r,np.random.beta(beta1,beta2,size=(1,m))])
r = np.vstack([r,np.random.exponential(exp1,size=(1,m))])
r = np.vstack([r,np.random.uniform(unid1,size=(1,m))])
r = np.vstack([r,np.random.normal(norm1,norm2,size=(1,m))])
r = np.vstack([r,np.random.geometric(geo1,size=(1,m))])



n = r.shape[0]

#np.random.seed(1234)

f = np.random.uniform(-2*np.pi,2*np.pi,size=(n,m))

S = r*(np.cos(f)+1j*np.sin(f))

# Standardize data
S = inv(np.diag(S.std(1))).dot(S)

# Mixing using complex mixing matrix A
A = rand(n,n)+1j*rand(n,n)
X = A.dot(S)

alg = 'deflation'#'parallel'

K,W,Shat,EG = cica.complex_FastICA(X,max_iter=40,algorithm=alg,\
                    n_components=n)


#Compute the SSE
absKAHW  = np.abs((K.dot(A)).conj().T.dot(W))

print '\nPermutation matrix: \n%s'%str(np.round(absKAHW))

maximum = absKAHW.max(0)
SSE     = ((absKAHW**2).sum(0)-maximum**2+(1-maximum)**2).sum()
SIR     = 10*log10(((absKAHW*1./maximum).sum(0)-1).mean())

print "\nSSE:%.4f"%SSE
print "\nSIR:%.4f"%SIR

span = 20
start=np.random.randint(m-span)


fig = plt.figure('fastICA_demo')
fig.clf()

ax1      = fig.add_subplot(121)
for j in xrange(n):
    ax1.plot(np.ma.masked_invalid(EG[j]),'.-',label='c_%i'%(j+1))
ax1.set_ylabel('E[G(|W.T*X|^2)]')
ax1.set_xlabel('iteration #')
plt.legend(loc='best')

ax2  = fig.add_subplot(222)
ax2.plot(np.abs(S[:,start:start+span]).T,lw=3,alpha=.2,color='k')
ax2.plot(np.abs(Shat[:,start:start+span]).T,'--',color='r')
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Time (a.u.)')

ax3  = fig.add_subplot(224)
ax3.plot(np.angle(S[:,start:start+span]).T,lw=3,alpha=.2,color='k')
ax3.plot(np.angle(Shat[:,start:start+span]).T,'--',color='b')
ax3.set_ylabel('Angle')
ax3.set_xlabel('Time (a.u.)')

plt.show()




