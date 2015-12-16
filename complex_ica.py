import pdb,os,time,warnings
import numpy as np
from math import log
from numpy.linalg import *
from numpy.random import rand

"""
Author: Alex Bujan
Adapted from: Ella Bingham, 1999

Original article citation:
Ella Bingham and Aapo Hyvaerinen, "A fast fixed-point algorithm for 
independent component analysis of complex valued signals", 
International Journal of Neural Systems, Vol. 10, No. 1 (February, 2000) 1-8

Original code url:
http://users.ics.aalto.fi/ella/publications/cfastica_public.m

Date: 12/11/2015
"""

def abs_sqr(W,X):
    return abs(W.conj().T.dot(X))**2

def complex_FastICA(X,epsilon=.1,algorithm='parallel',\
                    max_iter=100,tol=1e-4,whiten=True,\
                    w_init=None,n_components=None):
    """Performs Fast Independent Component Analysis of complex-valued signals
    
    Parameters
    ----------

    X: array, shape (n_features,n_samples)

    epsilon :  arbitrary constant in the contrast G function used in the
        approximation to neg-entropy.

    algorithm : {'parallel', 'deflation'}, optional
        Apply a parallel or deflational FASTICA algorithm.

    w_init : (n_components, n_components) array, optional
        Initial un-mixing array of dimension (n.comp,n.comp).
        If None (default) then an array of normal r.v.'s is used.

    tol: float, optional
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    max_iter : int, optional
        Maximum number of iterations.
    
    whiten : boolean, optional
        If True, perform an initial whitening of the data.
        If False, the data is assumed to be already white.

    n_components : int, optional
        Number of components to extract. If None, n_components = n_features.

    Returns
    -------

    W : array, shape (n_components, n_components)
        Estimated un-mixing matrix.

    K : array, shape (n_components, n_features) | None.
        If whiten is 'True', K is the pre-whitening matrix projecting the data
        onto the principal components. If whiten is 'False', K is 'None'.

    EG : array, shape(N-components,max_iter)
        Expectation of the contrast function E{G(|W'*X|^2)}. This array may be 
        padded with NaNs at the end.

    S : array, shape (n_samples, n_components)
        Estimated sources. If whiten the sources are projected back onto the 
        original signal space S = Sx W K X
    """

    n,m  = X.shape
    
    if n_components!=None:
        n = n_components

    if whiten:
        X-=X.mean(1,keepdims=True)
        Ux,Sx = eig(np.cov(X))
        K     = np.sqrt(inv(np.diag(Ux))).dot(Sx.conj().T)[:n]
        X     = K.dot(X)
        del Ux
    else:
        K = None

    EG = np.ones((n,max_iter))*np.nan

    if algorithm=='deflation':
        #un-mixing matrix
        W = np.zeros((n,n),dtype=np.complex)

        for k in xrange(n):
            if w_init!=None:
                w = w_init[:,k]
            else:
                w = np.random.normal(size=(n,1))+\
                1j*np.random.normal(size=(n,1))
            w/=norm(w)
            n_iter  = 0

            for i in xrange(max_iter):

                wold = np.copy(w)

                #derivative of the contrast function
                g  =  1./(epsilon+abs_sqr(w,X))
                #derivative of g
                dg = -1./(epsilon+abs_sqr(w,X))**2

                w  = (X*np.repeat(np.conj(w.conj().T.dot(X)),n,0)*\
                     np.repeat(g,n,0)).mean(1).reshape((n,1))-\
                     (g+abs_sqr(w,X)*dg).mean()*w

                del g,dg

                w/=norm(w)

                # Decorrelation
                w-=W.dot(W.conj().T.dot(w))
                w/=norm(w)

                EG[k,n_iter] = (np.log(epsilon+abs_sqr(w,X))).mean()

                n_iter+=1

                lim = (abs(abs(wold)-abs(w))).sum()
                if lim<tol:
                    break

            if n_iter==max_iter and lim>tol:
                warnings.warn('FastICA did not converge. Consider increasing '
                              'tolerance or the maximum number of iterations.')

            W[:,k] = w.ravel()

    elif algorithm=='parallel':
        #un-mixing matrix
        if w_init!=None:
            W = w_init
        else:
            W = np.random.normal(size=(n,n))+\
                1j*np.random.normal(size=(n,n))
        n_iter = 0
        #needed for decorrelation
        C = np.cov(X)

        for i in xrange(max_iter):

            for j in xrange(n):

                #derivative of the contrast function
                g  =  (1./(epsilon+abs_sqr(W[:,j],X))).reshape((1,m))
                #derivative of g
                dg = -(1./(epsilon+abs_sqr(W[:,j],X))**2).reshape((1,m))

                W[:,j]  = (X*np.repeat(np.conj(W[:,j].conj().T.dot(X)).reshape((1,m)),n,0)*\
                          np.repeat(g,n,0)).mean(1)-\
                          (g+abs_sqr(W[:,j],X)*dg).mean()*W[:,j]
                del g,dg

            # Symmetric decorrelation
            Uw,Sw = eig(W.conj().T.dot(C.dot(W)))
            W   = W.dot(Sw.dot(inv(np.sqrt(np.diag(Uw))).dot(Sw.conj().T)))
            del Uw,Sw

            EG[:,n_iter] = (np.log(epsilon+abs_sqr(W,X))).mean(1)

            n_iter+=1

    if whiten:
        S = Sx[:,:n].dot(W.conj().T.dot(X))
    else:
        S = W.conj().T.dot(X)

    return K,W,S,EG
