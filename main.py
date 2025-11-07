import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#график лог-длины интервала относительно n при фикс уровне доверия
#график лог-длины интервала относительно уровня доверия при фикс n

a = 11.666
sigma = 1.7

#for a with known sigma^2
def ci1(X, eps, sigma):
    mean = np.mean(X)
    tau = stats.norm.ppf( (1 + eps) / 2 )
    n = X.shape[0]
    return [ mean - tau * sigma / (n**0.5), mean + tau * sigma / (n**0.5) ]
    

#for sigma^2 with known a
def ci2(X, eps, a):
    n = X.shape[0]
    g2 = stats.chi2.ppf( (1 + eps) / 2, df=n)
    g1 = stats.chi2.ppf( (1 - eps) / 2, df=n)
    S_1_sq =  np.mean( (X - a) ** 2 )
    return [ n * S_1_sq / g2, n * S_1_sq / g1 ]


#for a with unknown sigma^2
def ci3(X, eps):
    mean = np.mean(X)
    tau = stats.t.ppf( (1 + eps) / 2, df=n-1)
    n = X.shape[0]
    S_0 = np.sqrt( np.mean((X -mean)**2) * (n / (n-1)) )
    return [ mean - tau * S_0 / (n**0.5), mean + tau * S_0 / (n**0.5) ]

#for sigma^2 with uknown a
def ci4(X, eps):
    n = X.shape[0]
    g2 = stats.chi2.ppf( (1 + eps) / 2, df=n-1)
    g1 = stats.chi2.ppf( (1 - eps) / 2, df=n-1)
    S_0 = np.sqrt(  np.mean((X -np.mean(X))**2) * (n / (n-1)) )
    return [ (n-1) * S_0 / g2, (n-1) * S_0 / g1 ]


