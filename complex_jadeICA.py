from __future__ import division
import pdb,os,time,warnings
import numpy as np
from math import log
from numpy.linalg import *
from numpy.random import rand
from scipy.linalg import sqrtm

def jade(X,m=None,max_iter=100,nem=None):
    """Source separation of complex signals via Joint Approximate 
        Diagonalization of Eigen-matrices or JADE

    Parameters
    ----------

    X : array, shape (n_features, n_samples)

    m : int, optional
        number of sources. If None, equals the number of features.

    Returns
    -------

    A : array, shape (n_components, n_features) 
        estimate of the mixing matrix
    S : array, shape (n_components, n_samples) 
        estimate of the source signals

    Original script in Matlab - version 1.6.  Copyright: JF Cardoso.
    Author: Alex Bujan <afbujan@gmail.com>
    Date: 20/01/2016
    """

    n,T     = X.shape

    if m==None:
        m = n

    if nem==None:
        nem = m

    # a statistical threshold for stopping joint diag
    tol = 1/(np.sqrt(T)*1e2)

    '''
    whitening
    '''

    X-=X.mean(1,keepdims=True)

    if m<n:
        #assumes white noise
        D,U     = eig((X.dot(X.conj().T))/T)
        k       = np.argsort(D)
        puiss   = D[k]
        ibl     = np.sqrt(puiss[-m:]-puiss[:-m].mean())
        bl      = 1/ibl
        W       = np.diag(bl).dot(U[:,k[-m:]].conj().T)
        IW      = U[:,k[-m:]].dot(np.diag(ibl))
    else:
        #assumes no noise
        IW      = sqrtm((X.dot(X.conj().T))/T)
        W       = inv(IW)

    Y    = W.dot(X)

    '''
    Cumulant estimation
    '''

    #covariance
    R    = Y.dot(Y.conj().T)/T
    #pseudo-covariance
    C    = Y.dot(Y.T)/T

    Q    = np.zeros((m*m*m*m,1),dtype=np.complex)
    idx  = 0

    for lx in xrange(m):
        Yl = Y[lx]
        for kx in xrange(m):
            Ykl = Yl*Y[kx].conj()
            for jx in xrange(m):
                Yjkl = Ykl*Y[jx].conj()
                for ix in xrange(m):
                    Q[idx] = Yjkl.dot(Y[ix].T)/T-\
                               R[ix,jx]*R[lx,kx]-\
                               R[ix,kx]*R[lx,jx]-\
                               C[ix,lx]*np.conj(C[jx,kx])
                    idx+=1

    '''
    computation and reshaping of the significant eigen matrices
    '''
    D,U = eig(Q.reshape((m*m,m*m)))
    K   = np.argsort(abs(D))
    la  = abs(D)[K]
    M   = np.zeros((m,nem*m),dtype=np.complex)
    h   = (m*m)-1
    for u in np.arange(0,nem*m,m):
        M[:,u:u+m] = la[h]*U[:,K[h]].reshape((m,m))
        h-=1

    '''
    joint approximate diagonalization of the eigen-matrices
    '''
    B       = np.array([[1,0,0],[0,1,1],[0,0,0]])+\
              1j*np.array([[0,0,0],[0,0,0],[0,-1,1]])
    Bt      = B.conj().T
    V       = np.eye(m).astype(np.complex)

    encore  = True

    #Main loop

    for n_iter in xrange(max_iter):

        for p in xrange(m-1):

            for q in np.arange(p+1,m):

                Ip = np.arange(p,nem*m,m)
                Iq = np.arange(q,nem*m,m)

                #Computing the Givens angles
                g       = np.vstack([M[p,Ip]-M[q,Iq],M[p,Iq],M[q,Ip]])
                D,vcp   = eig(np.real(B.dot(g.dot(g.conj().T)).dot(Bt)))
                K       = np.argsort(D)
                la      = D[K]

                angles  = vcp[:,K[2]]

                if angles[0]<0:
                    angles*=-1

                c    = np.sqrt(.5+angles[0]/2)
                s    = .5*(angles[1]-1j*angles[2])/c

                #updates matrices M and V by a Givens rotation
                if abs(s)>tol:
                    pair            = np.hstack((p,q))
                    G               = np.vstack(([c,-s.conj()],[s,c]))
                    V[:,pair]       = V[:,pair].dot(G)
                    M[pair,:]       = G.conj().T.dot(M[pair,:])
                    ids             = np.hstack((Ip,Iq))
                    M[:,ids]        = np.hstack((c*M[:,Ip]+s*M[:,Iq],\
                                                 -s.conj()*M[:,Ip]+c*M[:,Iq]))
                else:
                    encore          = False

        if encore==False:
            break

    if n_iter+1==max_iter:
        warnings.warn('JadeICA did not converge. Consider increasing '
                      'the maximum number of iterations.')

    '''
    estimation of the mixing matrix and sources
    '''
    A    = IW.dot(V)
    S    = V.conj().T.dot(Y)

    return A,S,W,V
