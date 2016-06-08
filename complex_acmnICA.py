from __future__ import division
import pdb
import numpy as np
from numpy.linalg import *
from math import log,gamma
from numpy.random import rand
from scipy.special import psi,polygamma


"""
Author: Alex Bujan
Adapted from:
    Mike Novey and T. Adali, "ADAPTABLE NONLINEARITY FOR COMPLEX 
    MAXIMIZATION OF NONGAUSSIANITY AND A FIXED-POINT ALGORITHM" in
    IEEE MLSP 2006.,Adaptable ICA algorithm based on complex
    generalized Gaussian distribution
Date: 12/11/2015
"""


def ACMNsym(X,model='circ',p_init=1.25,\
            max_iter=40,tol=1e-4,epsilon=.01):
    """
    ICA of a complex-valued signal
    
    Input:
        X     : vector of mixtures
        model : specifies MLE algorithm. Options:
                + noncirc uses MLE 
                  that has noncircular model
                + circ assumes circular and 
                  runs mutch faster
    """

    n,m     = X.shape

    #Whitening
    X-=X.mean(1,keepdims=True)
    Dx,Ex   = eig(np.cov(X))
    K       = np.sqrt(inv(np.diag(Dx))).dot(Ex.conj().T)
    X       = K.dot(X)

    #cache the pseudo-covariance
    pC      = X.dot(X.T)/m

    #initialized shape parameter 'p' for each source
    params = np.repeat(p_init,n)

    '''
    FIXED POINT ALGORITHM
    '''

    #initialize the unmixing matrix
    W = np.random.normal(size=(n,n))+\
                    1j*np.random.normal(size=(n,n))

    for k in xrange(max_iter):

        Wold = np.copy(W)


        for kk in xrange(n): 

            '''
            1) Calculate current source estimate y = WHX
            '''

            y = W[:,kk].reshape((n,1)).conj().T.dot(X)

            '''
            2) Update W
            '''

            if model=='noncirc':

                p = params[kk]

            elif model=='circ':

                p = params[kk]/2

            abs_y    = abs(y)**2

            u       = abs_y + epsilon

            u1      = p * u**(p-1)
            u2      = p * (p-1) * u**(p-2)

            gRad    = ((u1*y.conj())*X).mean(1)
            ggg     = (u2*abs_y+u1).mean()
            B       = (u2*y.conj()**2).mean()*pC

            W[:,kk] = Wold[:,kk]*ggg - gRad + B.dot(Wold[:,kk].conj())

            '''
            3) Estimate p
            '''

            if model=='noncirc':

                aug_y    = np.concatenate([y.T,y.conj().T],axis=1).conj().T

                params[kk] = estimateGGDCovShapeIn(aug_y,params[kk])

            elif model=='circ':

                # Newton estimate of p

                p     = params[kk]

                abs_y = abs(y)

                u     = abs_y + epsilon

                up    = u**p

                sigP  = (abs_y**p).mean()**(1/p)

                gp    = -(1/p**2) * log(p) + (1/p**2) - \
                          psi(1+1/p) * (1/p**2) + \
                          ( (1/(sigP**p*p)) * up * (np.log(u) - \
                          ((1/p) + log(sigP)) ) ).mean()

                ggp   = 2*(1/p**3)*log(p) - 3*(1/p**3) + \
                          float(polygamma(1,1+1/p))*(1/p**4) + \
                          2*psi(1+1/p)*(1/p**3) + \
                          ( (1/(p*sigP**p)) * up * \
                          ( np.log(u)**2 - 2*(1/p)*np.log(u) - \
                          2*log(sigP)*np.log(u) + \
                          (2*(1/p**2) + 2*log(sigP)*(1/p) + \
                          log(sigP)**2) ) ).mean()

                p-=(gp/ggp)
                p = np.maximum(p,.2)
                p = np.minimum(p,3)

                params[kk] = p

        '''
        4) Normalize W (symmetric decorrelation)
        '''

        D,E = eig(W.conj().T.dot(W))
        W   = W.dot(E.dot(inv(np.sqrt(np.diag(D))).dot(E.conj().T)))

        '''
        5) Check for convergence
        '''

        if (abs(abs(Wold)-abs(W))).sum() < tol:
            break

    S = W.conj().T.dot(X)
    A = inv(K).dot(W)

    return A,S,K,W

def estimateGGDCovShapeIn(X,p_init):

    N = X.shape[1]
    R = np.cov(X)

    #start at Gaussian
    bestC       = p_init
    c           = bestC

    Rold        = np.zeros((2,2),dtype=np.complex)

    xRxC        = 0
    dirXRX      = 0
    dirXRX2     = 0

    for n in xrange(N):
        temp    = X[:,n].conj().T.dot(inv(R)).dot(X[:,n])
        xRxC    += (temp**c).real
        dirXRX  += (log(temp)*temp**c).real
        dirXRX2 += (log(temp)**2*temp**c).real

    c2  = gamma(2*1/c)/(2*gamma(1/c))

    c2p = log(c2) - (1/c) * 2*psi(2*1/c) - psi(1/c)

    gc  = N * ( (1/c) - (1/c**2) * 2*psi(2*1/c) + \
          (1/c**2) * 2*psi(1/c) ) - \
          (c2**c) * (c2p*xRxC + dirXRX)

    ##Second dir
    A   = N * ( (4*psi(2*1/c)/c**3) + \
          (4*polygamma(1,2*1/c)/c**4) - \
          (1/c**2) - (4*psi(1/c)/c**3) - \
          (2*polygamma(1,1/c)/c**4) )

    #Dir c2**c
    dc2C = log(c2)*(c2**c) - \
           c*(c2**(c-1))*(c2*2*psi(2*1/c)/c**2 - \
           c2*psi(1/c)/c**2)

    dc2p = -((psi(1/c) - 2*psi(2*1/c))/c**2) - \
            ((polygamma(1,1/c) - 4 * polygamma(1,2*1/c))/c**3)-\
            ((2*psi(2*1/c)/c**2) - psi(1/c)/c**2)

    B = dc2C*c2p*xRxC + c2**c * (dc2p*xRxC + c2p*dirXRX)

    C = dc2C*dirXRX + c2**c * dirXRX2

    ggc     = A-B-C
    cold    = c
    cn      = c - (1/ggc) * gc

    #Newton update with no negatives
    c       = np.minimum(4,np.maximum(.05,cn))

    return c

