"""Utility funcitons.
"""

import math

import numpy as np
import numpy.random as npr
import ot
from scipy.stats import ortho_group


##############################################################################
# Synthetic data
##############################################################################

def generate_data(size, dims, dist, pars):
    if dist == 'normal':
        data = npr.multivariate_normal(pars[0], pars[1], size=size)
        X = data[:, :dims[0]]
        Y = data[:, dims[0]:]
    elif dist == 'exp-normal':
        X = npr.exponential(pars[0], size).reshape(-1, 1)
        Y = []
        for x in X:
            Y.append(npr.multivariate_normal(
                (x[0] - pars[0])*pars[1], pars[2], size=1)[0])
        Y = np.asarray(Y)
    elif dist == 'normal-linear':
        X = npr.normal(size=(size, dims[0]))
        Y = (X[:, 0] + npr.normal(size=size)).reshape(-1, 1)
    elif dist == 'normal-sine':
        X = npr.normal(size=(size, dims[0]))
        Y = 20*np.sin(4*math.pi*(X[:, 0]**2 + X[:, 1]**2)) + npr.normal(size=size)
        Y = Y.reshape(-1, 1)
    elif dist == 'normal-sign':
        X = npr.normal(size=(size, dims[0]))
        Y = np.abs(npr.normal(size=size)) * np.prod(np.sign(X), axis=1)
        Y = Y.reshape(-1, 1)
    else:
        x = generate_ica_source(size, dist)
        y = generate_ica_source(size, dist)
        X, Y = prepare_synthetic_data(x, y, pars, dims)
    return X, Y


def generate_ica_source(size, dist):
    """Generate source data from ICA benchmark densities in [Gretton et al. '05].

    Parameters
    ----------
    size : int
        Sample size.
    dist : string
        Density label.
    """
    if dist == 'g':  # symmetric mixture of Gaussians
        mu = (1 - 0.15/2)**(1/4)
        sigma = (1 - mu**2)**0.5
        hidden = npr.binomial(1, 0.5, size=size)
        data = np.empty((size,))
        pos = hidden == 1
        data[pos] = npr.normal(mu, sigma, np.sum(pos, dtype=np.int64))
        neg = hidden == 0
        data[neg] = npr.normal(-mu, sigma, np.sum(neg, dtype=np.int64))
    elif dist == 'e':  # shifted exponential
        data = npr.exponential(size=size) - 1
    else:
        raise ValueError('Invalid distribution name.')
    return data


def prepare_synthetic_data(x, y, theta, dims):
    a = math.cos(theta) * x - math.sin(theta) * y
    b = math.sin(theta) * x + math.cos(theta) * y
    size = len(x)
    a = np.hstack((a.reshape(-1, 1), npr.normal(size=(size, dims[0]-1))))
    b = np.hstack((b.reshape(-1, 1), npr.normal(size=(size, dims[1]-1))))
    return np.dot(a, ortho_group.rvs(dims[0])), np.dot(b, ortho_group.rvs(dims[1]))


##############################################################################
# Real data
##############################################################################\

def load_data(fname, n=64):
    data = np.loadtxt(fname)
    size = int(data.shape[0] / 4)
    out = [(data[:n], data[i*size:i*size+n]) for i in range(1,4)]
    return out


##############################################################################
# Independence tests
##############################################################################

def median_dist(X, Y):
    xdists = ot.dist(X, X)[np.triu_indices(len(X))]
    ydists = ot.dist(Y, Y)[np.triu_indices(len(Y))]
    return np.median(xdists[xdists > 0]), np.median(ydists[ydists > 0])


def default_reg(X, Y):
    xdists = ot.dist(X, X)
    ydists = ot.dist(Y, Y)
    return (np.mean(xdists) + np.mean(ydists)) / math.log(len(X) * len(Y))


def normalize(X, Y):
    xmed, ymed = median_dist(X, Y)
    X = X / xmed**0.5
    Y = Y / ymed**0.5
    return X, Y
