import pdb
import numpy as np
from math import log
from numpy.linalg import *
from numpy.random import rand
import matplotlib.pyplot as plt
plt.ion()

"""
Author: Alex Bujan
Adapted from: Ella Bingham, 1999

Orinal article citation:
Ella Bingham and Aapo Hyvärinen, "A fast fixed-point algorithm for 
independent component analysis of complex valued signals", 
International Journal of Neural Systems, Vol. 10, No. 1 (February, 2000) 1-8

Original code url:
http://users.ics.aalto.fi/ella/publications/cfastica_public.m

Date: 12/11/2015
"""

def demo():
    m = 50000
    n = 5
    exp1 = np.ceil(10*rand())
    r = np.random.exponential(exp1,size=(n,m))
    f = np.zeros(r.shape)
    for j in xrange(n):
        f[j] = np.random.uniform(-2*np.pi,2*np.pi,size=(1,m))
    s = r*np.cos(f)+1j*np.sin(f)

    W,shat,SSE = complex_FastICA(s,eps=.1,defl=False,plot=True)

def abs_sqr(w,x):
    return np.abs(w.conj().T.dot(x))**2


def complex_FastICA(X,eps=.1,defl=False,\
                    plot=False,center=False,\
                    maxcounter=30):
    """
    eps :  epsilon in G
    defl:  when True components are estimated one by one (deflationary mode) 
           otherwise all components are estimated simultaneously (default).
    """
    n,m  = X.shape

    '''
    Signal whitening
    '''
    X    = inv(np.diag(X.std(1))).dot(X)

    # Mixing using complex mixing matrix A
    A    = rand(n,n)+1j*rand(n,n)
    xold = A.dot(X)

    '''
    Whitening of x
    '''
    Dx,Ex = eig(np.cov(xold))
    Q     = np.sqrt(inv(np.diag(Dx))).dot(Ex.conj().T)
    x     = Q.dot(xold)
    if center:
        x-=x.mean(1,keepdims=True)


    EG = np.ones((n,maxcounter))*np.nan

    '''
    FIXED POINT ALGORITHM
    '''
#    Components estimated one by one
    if defl:
        W           = np.zeros((n,n),dtype=np.complex)

        for k in xrange(n):

            w           = rand(n,1)+1j*rand(n,1)
            wold        = np.zeros((n,1),dtype=np.complex)
            counter     = 0

            while (np.abs(np.abs(wold)-np.abs(w))).sum()>1e-4 and counter<maxcounter:

                wold = np.copy(w)

                g    =  1./(eps+abs_sqr(w,x))
                dg   = -1./(eps+abs_sqr(w,x))**2

                w    = (x*np.repeat(np.conj(w.conj().T.dot(x)),n,0)*\
                       np.repeat(g,n,0)).mean(1).reshape((n,1))-\
                       (g+abs_sqr(w,x)*dg).mean()*w

                w/=norm(w)

                # Decorrelation
                w-=W.dot(W.conj().T.dot(w))
                w/=norm(w)

                EG[k,counter] = (np.log(eps+abs_sqr(w,x))).mean()

                counter+=1

            if np.isnan(EG[k,counter-1]-EG[k,counter-2]):
                break

            W[:,k] = w.reshape((n,))

#    symmetric approach all components estimated simultaneously
    else:
        C          = np.cov(x)
        W          = np.random.normal(size=(n,n))+\
                     1j*np.random.normal(size=(n,n))
        counter    = 0
        while counter<maxcounter:

            for j in xrange(n):

                gWx  =  (1./(eps+abs_sqr(W[:,j],x))).reshape((1,m))
                dgWx = -(1./(eps+abs_sqr(W[:,j],x))**2).reshape((1,m))

                W[:,j]  = (x*np.repeat(np.conj(W[:,j].conj().T.dot(x)).reshape((1,m)),n,0)*\
                          np.repeat(gWx,n,0)).mean(1)-\
                          (gWx+abs_sqr(W[:,j],x)*dgWx).mean()*W[:,j]

            # Symmetric decorrelation
            D,E = eig(W.conj().T.dot(C.dot(W)))
            W   = W.dot(E.dot(inv(np.sqrt(np.diag(D))).dot(E.conj().T)))

            EG[:,counter] = (np.log(eps+abs_sqr(W,x))).mean(1)

            counter+=1

    absQAHW = np.abs((Q.dot(A)).conj().T.dot(W))
    maximum = np.max(absQAHW)
    SSE      = (np.sum(absQAHW**2)-maximum**2+np.repeat(1-maximum,5)**2).sum()
    shat = W.conj().T.dot(x)

    if plot:
        fig = plt.figure('demo')
        ax = fig.add_subplot(111)
        ax.plot(np.ma.masked_invalid(EG.T),'.-')
        ax.set_title('Convergence of G')
        plt.show()

    return W,shat,SSE

if __name__=='__main__':
    demo()
