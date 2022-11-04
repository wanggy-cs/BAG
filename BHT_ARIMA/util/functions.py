import numpy as np
import pandas as pd
import tensorly.backend as T
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.base import unfold, fold
from tensorly.tenalg import multi_mode_dot, mode_dot
from scipy import linalg
from BHT_ARIMA.util.svd import svd_fun


def svd_init(tensor, modes, ranks):
    factors = []
    for index, mode in enumerate(modes):
        eigenvecs, _, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=ranks[index])
        factors.append(eigenvecs)
    return factors


def init(dims, ranks):
    factors = []
    for index, rank in enumerate(ranks):
        U_i = np.zeros((rank, dims[index]))
        mindim = min(dims[index], rank)
        for i in range(mindim):
            U_i[i][i] = 1
        factors.append(U_i)
    return factors


def autocorr(Y, lag=10):
    T = len(Y)
    r = []
    for l in range(lag+1):
        product = 0
        for t in range(T):
            tl = l - t if t < l else t - l
            product += np.sum(Y[t] * Y[tl])
        r.append(product)
    return r


def fit_ar(Y, p=10):
    r = autocorr(Y, p)
    R = linalg.toeplitz(r[:p])
    r = r[1:]
    A = linalg.pinv(R).dot(r)
    return A

def fit_ar_ma(Y,p=10,q=1):
    N = len(Y)
    
    A = fit_ar(Y, p)
    B = [0.]
    if q>0:
        Res = []
        for i in range(p,N):
            res = Y[i] - np.sum([ a * Y[i-j] for a, j in zip(A, range(1, p+1))], axis=0)
            Res.append(res)
        B = fit_ar(Res, q)
    return A, B



