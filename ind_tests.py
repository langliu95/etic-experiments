"""Test statistics.

Author: Lang Liu
"""

from __future__ import absolute_import, division, print_function

import os
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import ot
from scipy.special import lambertw

class NonparIndependenceTest(object):
    """A class for nonparametric independence tests.
    """
    
    def __init__(self):
        raise NotImplementedError

    def compute_stat(self, xmat, ymat):
        raise NotImplementedError
    
    def decision(self, xmat, ymat, alpha, nperms, parallel=False, ncores=1):
        size = xmat.shape[0]
        stat = self.compute_stat(xmat, ymat)
        
        # compute p-value
        def permute_stat(repeat):
            np.random.seed()
            ind = np.random.choice(size, size=size, replace=False)
            return self.compute_stat(xmat, ymat[np.ix_(ind, ind)])

        if parallel:
            cores = min(ncores, os.cpu_count())
            with Pool(cores) as pool:
                stats = pool.map(permute_stat, range(nperms))
        else:
            stats = list(map(permute_stat, range(nperms)))
        pval = np.mean(np.asarray(stats) > stat)
        return int(pval < alpha)


##########################################################################
# HSIC
##########################################################################

# For bootstrap approach
def hsic(xgram, ygram):
    """Compute the V-statistic version of HSIC.

    Parameters
    ----------
    xgram : Gram matrix of Xs.
    ygram : Gram matrix of Ys.

    Returns
    -------
    stat : float
        The HISC statistic.
    """
    term1 = np.mean(xgram * ygram)
    term2 = np.mean(xgram) * np.mean(ygram)
    term3 = np.mean(np.mean(xgram, axis=1) * np.mean(ygram, axis=1))
    return term1 + term2 - 2*term3

    
class HSICTest(NonparIndependenceTest):
    """A class for independence testing with HSIC.
    """
    
    def __init__(self):
        pass
    
    def compute_stat(self, xmat, ymat):
        stat = hsic(xmat, ymat)
        return stat


##########################################################################
# ETIC (eot-based independence criterion)
##########################################################################

def _sanwich(u, M1, M2):
    return np.dot(M1, np.dot(np.dot(np.dot(M1.T, u), M2), M2.T))


def get_random_feature(X, nfeat, reg, cst=None):
    if cst is None:
        cst = np.max(np.sum(X**2, axis=1))
    dim = X.shape[1]
    y = cst / (reg * dim)
    q = np.real(0.5 * np.exp(lambertw(y)))
    sigma2 = q * reg / 4
    U = np.random.multivariate_normal(np.zeros(dim), sigma2 * np.eye(dim), nfeat)
    exponent = -(2 * ot.dist(X, U)) / reg + np.sum(U**2, axis=1) / reg / q
    out = (2*q)**(dim/4) * np.exp(exponent) / nfeat**0.5
    return out


def eot(cmat, reg, marginals=None, cost_only=False):
    """Solve the EOT problem.

    Parameters
    ----------
    cmat : numpy.array (2D)
        Cost matrix.
    reg : float
        Regularization parameter.
    """
    size = cmat.shape
    # uniform distributions on the sample
    if marginals is None:
        a = np.ones(size[0]) / size[0]
        b = np.ones(size[1]) / size[1]
    else:
        a, b = marginals
    sol = ot.sinkhorn(a, b, cmat, reg)
    cost = np.sum(sol * cmat)
    if not cost_only:
        cost += reg * np.sum(sol * np.log(sol / np.outer(a, b)))
    return cost, sol
    

def sinkhorn_grid(M1, M2, reg, a=None, b=None, numItermax=1000, stopThr=1e-9,
                  low_rank=False, transport=False, cost_only=False,
                  verbose=False, log=False):
    """Solve the EOT problem for additive costs.

    Parameters
    ----------
    M1 : numpy.array (2D)
        Cost matrix (or its low rank approximation) of the first argument.
    M2 : numpy.array (2D)
        Cost matrix (or its low rank approximation) of the second argument.
    reg : float
        Regularization parameter.
    a : numpy.array (2D), optional
        Left marginal distribution, by default None.
    b : numpy.array (2D), optional
        Right marginal distribution, by default None.
    numItermax : int, optional
        Maximum number of iterations, by default 1000.
    stopThr : float, optional
        Stopping threshold for the violation of marginals, by default 1e-9.
    low_rank : bool, optional
        Indication of the `M1` and `M2` being the low rank approximations
        of the cost matrices, by default False.
    transport : bool, optional
        Indication of returning the entropic optimal transport map,
        by default False.
    cost_only : bool, optional
        Indication of returning the cost without the regularization part,
        by default False.
    verbose : bool, optional
        Indication of printing performance metrics during the iteration process,
        by default False.
    log : bool, optional
        Indication of returning all intermediate results, by default False.

    Returns
    -------
    cost : float
        Entropic cost.
    log : dictionary
        Intermediate results during the iteration process. Return it only if
        `log` is `True`.
    tran : numpy.array (2D)
        Entropic optimal transport map. Return it only if `transport` is `True`.
    """
    n, m = M1.shape[0], M2.shape[0]

    if a is None:
        a = np.full((n, m), 1.0/n/m, dtype=M1.dtype)
    if b is None:
        b = np.full((n, m), 1.0/n/m, dtype=M1.dtype)

    if log:
        log = {'err': []}

    u = np.ones((n, m)) / n / m
    v = np.ones((n, m)) / n / m

    if not low_rank:
        K1 = np.exp(M1 / (-reg))
        K2 = np.exp(M2 / (-reg))

    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        if low_rank:
            KtransposeU = _sanwich(u, M1, M2)
            v = b / KtransposeU
            u = a / _sanwich(v, M1, M2)
        else:
            KtransposeU = np.dot(np.dot(K1.T, u), K2)
            v = b / KtransposeU
            u = a / np.dot(np.dot(K1, v), K2.T)

        if (np.any(KtransposeU == 0)
                or np.any(np.isnan(u)) or np.any(np.isnan(v))
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # compute right marginal tmp= (diag(u)Kdiag(v))^T1
            if low_rank:
                tmp = v * _sanwich(u, M1, M2)
            else:
                tmp = v * np.dot(np.dot(K1.T, u), K2)

            err = np.linalg.norm(tmp - b)  # violation of marginal
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    if log:
        log['u'] = u
        log['v'] = v

    if transport:  # return OT matrix
        if low_rank:
            K1 = np.dot(M1, M1.T)
            K2 = np.dot(M2, M2.T)
        K = np.repeat(K1, m, axis=0)
        K = np.repeat(K, m, axis=1)
        K *= np.tile(K2, (n, n))
        tran = u.reshape((-1, 1)) * K * v.reshape((1, -1))
        if log:
            return tran, log
        else:
            return tran
    else:
        if low_rank:
            tmp1 = np.dot(np.dot(u, M2), np.dot(M2.T, v.T))
            tmp2 = np.dot(np.dot(u.T, M1), np.dot(M1.T, v))
            K1 = np.dot(M1, M1.T)
            M1 = -reg*np.log(K1)
            K2 = np.dot(M2, M2.T)
            M2 = -reg*np.log(K2)
        else:
            tmp1 = np.dot(np.dot(u, K2), v.T)
            tmp2 = np.dot(np.dot(u.T, K1), v)
        cost = np.sum(K1 * M1 * tmp1) + np.sum(K2 * M2 * tmp2)
        if cost_only:  # no regularization part
            if log:
                return cost, log
            else:
                return cost
        else:
            cost += reg * np.sum(K1 * np.log(K1) * tmp1) + reg * np.sum(
                K2 * np.log(K2) * tmp2)
            aind, bind = a != 0, b != 0
            cost += reg*np.sum(a[aind]*np.log(u[aind]/a[aind])) + \
                reg*np.sum(b[bind]*np.log(v[bind]/b[bind]))
            if log:
                return cost, log
            else:
                return cost


class ETICTest(NonparIndependenceTest):
    """A class for independence testing with ETIC.
    
    It is only implemented for continuous random variables.
    """
    
    def __init__(self, eps):
        self.eps = eps
    
    def compute_stat(self, xmat, ymat, low_rank=False):
        size = xmat.shape[0]
        cost = sinkhorn_grid(
            xmat, ymat, self.eps, a=np.identity(size)/size, low_rank=low_rank)
        cost2 = sinkhorn_grid(xmat, ymat, self.eps, low_rank=low_rank)
        if low_rank:
            xgram = np.dot(xmat, xmat.T)
            if np.any(xgram < 0):
                xgram = np.abs(xgram)
            ygram = np.dot(ymat, ymat.T)
            if np.any(ygram < 0):
                ygram = np.abs(ygram)
            
            cost1, _ = eot(
                -self.eps*np.log(xgram*ygram),
                self.eps)
        else:
            cost1, _ = eot(xmat + ymat, self.eps)
        return cost - cost1/2 - cost2/2
    
    def decision_with_rf(self, xmat, ymat, alpha, nperms, parallel=False, ncores=1):
        size = xmat.shape[0]
        stat = self.compute_stat(xmat, ymat, low_rank=True)
        
        # compute p-value
        def permute_stat(repeat):
            np.random.seed()
            ind = np.random.choice(size, size=size, replace=False)
            return self.compute_stat(xmat, ymat[ind, :], low_rank=True)

        if parallel:
            cores = min(ncores, os.cpu_count())
            with Pool(cores) as pool:
                stats = pool.map(permute_stat, range(nperms))
        else:
            stats = list(map(permute_stat, range(nperms)))
        pval = np.mean(np.asarray(stats) > stat)
        return int(pval < alpha)
    
    
class AdaptiveETICTest(NonparIndependenceTest):
    """A class for independence testing with adaptive ETIC.
    
    It is only implemented for continuous random variables.
    """
    
    def __init__(self, eps):
        self.eps = eps
    
    def _compute_stat(self, xmat, ymat, eps, low_rank=False):
        size = xmat.shape[0]
        cost = sinkhorn_grid(
            xmat, ymat, eps, a=np.identity(size)/size, low_rank=low_rank)
        cost2 = sinkhorn_grid(xmat, ymat, eps, low_rank=low_rank)
        if low_rank:
            cost1, _ = eot(
                -eps*np.log(np.dot(xmat, xmat.T)*np.dot(ymat, ymat.T)), eps)
        else:
            cost1, _ = eot(xmat + ymat, eps)
        return cost - cost1/2 - cost2/2
    
    def _compute_normalized_stat(self, xmat, ymat, eps, low_rank=False):
        stat = self._compute_stat(xmat, ymat, eps, low_rank)
        size = len(xmat)
        stats = []
        for _ in range(20):
            ind = np.random.choice(size, size=size, replace=False)
            stats.append(self._compute_stat(
                xmat, ymat[np.ix_(ind, ind)], eps, low_rank))
        return (stat - np.mean(stats)) / np.std(stats)
    
    def compute_stat(self, xmat, ymat, low_rank=False):
        stats = []
        for eps in self.eps:
            stats.append(self._compute_normalized_stat(
                xmat, ymat, eps, low_rank))
        return max(stats)
    
    def decision_with_rf(self, xmat, ymat, alpha, nperms, parallel=False, ncores=1):
        size = xmat.shape[0]
        stat = self.compute_stat(xmat, ymat, low_rank=True)
        
        # compute p-value
        def permute_stat(repeat):
            np.random.seed()
            ind = np.random.choice(size, size=size, replace=False)
            return self.compute_stat(xmat, ymat[ind, :], low_rank=True)

        if parallel:
            cores = min(ncores, os.cpu_count())
            with Pool(cores) as pool:
                stats = pool.map(permute_stat, range(nperms))
        else:
            stats = list(map(permute_stat, range(nperms)))
        pval = np.mean(np.asarray(stats) > stat)
        return int(pval < alpha)


##########################################################################
# Baseline tests
##########################################################################

def equal_partition(X, nparts):
    n, dim = X.shape
    idx = np.arange(n)
    prevblock = [n]
    for d in range(dim):
        numblk = len(prevblock)
        start = 0
        b_end = np.cumsum(prevblock)
        tmpblock = []
        
        for i in range(numblk):
            subidx = idx[start:b_end[i]]
            order = np.argsort(X[subidx, d])
            nk = len(subidx)
            num = int(nk / nparts)
            if num <= 1:
                raise ValueError("Use smaller number of partitions.")
            subblk = list(np.array([num]*nparts) + np.array(
                [0]*(nparts-nk%nparts) + [1]*(nk%nparts)))
            tmpblock += subblk
            idx[start:b_end[i]] = subidx[order]
            start = b_end[i]
        prevblock = tmpblock
    return idx, tmpblock


class InfoIndependenceTest(object):
    """A class for independence tests based on information divergences.
    """
    
    def __init__(self):
        raise NotImplementedError

    def compute_stat(self, X, Y, nparts, partX=None, partY=None):
        raise NotImplementedError
    
    def decision(self, X, Y, alpha, nperms, nparts=2, parallel=False, ncores=1):
        size = X.shape[0]
        # equal partition
        idxX, blockX = equal_partition(X, nparts)
        blockX = np.cumsum(blockX)
        idxY, blockY = equal_partition(Y, nparts)
        blockY = np.cumsum(blockY)
        stat = self.compute_stat(X, Y, nparts, [idxX, blockX], [idxY, blockY])
        
        # compute p-value
        def permute_stat(repeat):
            np.random.seed()
            ind = np.random.choice(size, size=size, replace=False)
            inv_ind = np.argsort(ind)
            return self.compute_stat(
                X, Y[ind], nparts, [idxX, blockX], [inv_ind[idxY], blockY])

        if parallel:
            cores = min(ncores, os.cpu_count())
            with Pool(cores) as pool:
                stats = pool.map(permute_stat, range(nperms))
        else:
            stats = list(map(permute_stat, range(nperms)))
        pval = np.mean(np.asarray(stats) > stat)
        return int(pval < alpha)
    

class L1Test(InfoIndependenceTest):
    """A class for independence testing with the L1 distance.
    """
    
    def __init__(self):
        pass
    
    def compute_stat(self, X, Y, nparts, partX=None, partY=None):
        n = X.shape[0]
        dx, dy = X.shape[1], Y.shape[1]
        if partX is None:
            idxX, blockX = equal_partition(X, nparts)
            blockX = np.cumsum(blockX)
        else:
            idxX, blockX = partX
        if partY is None:
            idxY, blockY = equal_partition(Y, nparts)
            blockY = np.cumsum(blockY)
        else:
            idxY, blockY = partY
        stat = 0.0
        
        for ix in range(nparts**dx):
            for iy in range(nparts**dy):
                indices = np.intersect1d(
                    idxX[blockX[ix]-blockX[0]:blockX[ix]],
                    idxY[blockY[iy]-blockY[0]:blockY[iy]])
                count = len(indices)
                stat += np.abs(count/n - 1/nparts**(dx+dy))
        return stat
    

class MutualInfoTest(InfoIndependenceTest):
    """A class for independence testing with the mutual information.
    """
    
    def __init__(self):
        pass
    
    def compute_stat(self, X, Y, nparts, partX=None, partY=None):
        n = X.shape[0]
        dx, dy = X.shape[1], Y.shape[1]
        if partX is None:
            idxX, blockX = equal_partition(X, nparts)
            blockX = np.cumsum(blockX)
        else:
            idxX, blockX = partX
        if partY is None:
            idxY, blockY = equal_partition(Y, nparts)
            blockY = np.cumsum(blockY)
        else:
            idxY, blockY = partY
        stat = 0.0
        
        for ix in range(nparts**dx):
            for iy in range(nparts**dy):
                indices = np.intersect1d(
                    idxX[blockX[ix]-blockX[0]:blockX[ix]],
                    idxY[blockY[iy]-blockY[0]:blockY[iy]])
                count = len(indices)
                if count > 0:
                    stat += 2*count/n*np.log(count/n * nparts**(dx+dy))
        return stat
