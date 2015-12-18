import pdb,os,time
import numpy as np
from math import log
from numpy.linalg import *
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import complex_ica as cica
reload(cica)
import complex_ica_jade as cicaj
reload(cicaj)

plt.ion()


m = 50000
n = 5
exp1 = np.ceil(10*rand())
r = np.random.exponential(exp1,size=(n,m))
f = np.zeros(r.shape)
for j in xrange(n):
    f[j] = np.random.uniform(-2*np.pi,2*np.pi,size=(1,m))
Xu = r*np.cos(f)+1j*np.sin(f)
# Standardize data
Xu = inv(np.diag(Xu.std(1))).dot(Xu)



# Mixing using complex mixing matrix A
A  = rand(n,n)+1j*rand(n,n)
Xm = A.dot(Xu)

alg = 'parallel'#'deflation'
K,W,S,EG = cica.complex_FastICA(Xm,max_iter=30,algorithm=alg,\
                    n_components=n)

#Compute the SSE

absKAHW  = np.abs((K.dot(A)).conj().T.dot(W))
maximum  = np.max(absKAHW)
SSE      = (np.sum(absKAHW**2)-maximum**2+np.repeat(1-maximum,5)**2).sum()

print SSE

ntp=20
fig = plt.figure('demo')
fig.clf()

ax      = fig.add_subplot(121)
for j in xrange(n):
    ax.plot(np.ma.masked_invalid(EG[j]),'.-',label='c_%i'%(j+1))
ax.set_ylabel('E[G(|W.T*X|^2)]')
ax.set_xlabel('iteration #')
plt.legend(loc='best')

ax2  = fig.add_subplot(222)
ax2.plot(np.abs(Xu[:,:ntp].T),lw=3,alpha=.2,color='k')
ax2.plot(np.abs(S[:,:ntp].T),'--',color='r')
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Time (a.u.)')

ax2  = fig.add_subplot(224)
ax2.plot(np.angle(Xu[:,:ntp]).T,lw=3,alpha=.2,color='k')
ax2.plot(np.angle(S[:,:ntp]).T,'--',color='b')
ax2.set_ylabel('Angle')
ax2.set_xlabel('Time (a.u.)')

plt.show()




