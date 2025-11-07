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


dir = 'aKs'

def plot_len_vs_n(epsilons):
    for eps in epsilons:
        lengths = []
        sizes = range(7_000,25_000,10)
        for n in sizes:
            X = np.random.normal(a, sigma, n)
            l,r = ci1(X, eps, sigma)
            lengths.append(r - l)
        plt.plot(sizes, lengths, label=f'eps={eps}')
    plt.legend(loc='upper right')
    plt.title(f"len_VS_size")
    plt.xlabel('sample size')
    plt.ylabel('CI len')
    plt.savefig(f"./{dir}/fixed_eps/first.png")
    plt.close()


def plot_len_vs_eps(sizes,title):
    for n in sizes:
        lengths = []
        epsilons = np.linspace(0.1, 0.98, 20, endpoint=True)
        for eps in epsilons:
            X = np.random.normal(a, sigma, n)
            l,r = ci1(X, eps, sigma)
            lengths.append(r - l)
        plt.plot(epsilons, lengths, label=f'n={n}')
    plt.legend(loc='upper right')
    plt.title(f"len_VS_eps")
    plt.xlabel('eps')
    plt.ylabel('CI len')
    plt.savefig(f"./{dir}/fixed_n/{title}.png")
    plt.close()

epsilons = [0.8, 0.9, 0.92, 0.95, 0.98]
sizes1 = [10, 50, 100, 300, 500, 700, 1000, 3000]
sizes2 = [5000, 7000, 10_000, 15_000, 20_000, 25_000]
# plot_len_vs_eps(sizes1, "small_n")
# plot_len_vs_eps(sizes2, "big_n")
plot_len_vs_n(epsilons)
