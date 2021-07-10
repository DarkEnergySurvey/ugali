#!/usr/bin/env python
"""
Tests of ugali stats
"""
import numpy as np
import scipy.stats

import ugali.utils.stats

def generate_distribution(p1=[1.0,0.1,10000],p2=[-1.0,0.6,30000]):
    """ Create a bimodal distribution for testing. """
    mu1,sig1,num1 = p1
    mu2,sig2,num2 = p2
    g1 = scipy.stats.norm(mu1,sig1).rvs(num1)
    g2 = scipy.stats.norm(mu2,sig2).rvs(num2)

    samples = np.concatenate([g1,g2])
    return samples

def plot_intervals():
    import pylab as plt

    np.random.seed(12345)
    data = generate_distribution()

    funcs = [
        [ugali.utils.stats.peak_interval,'r'],
        [ugali.utils.stats.min_interval,'g'],
        [ugali.utils.stats.median_interval,'b'],
        [ugali.utils.stats.mean_interval,'m'],
    ]

    num,edges,_ = plt.hist(data,bins=100,histtype='stepfilled',color='gray')
    for i,(func,color) in enumerate(funcs):
        x,[x1,x2] = func(data,alpha=0.32)
        y = [(1-float(i)/len(funcs)) * num.max()]
        plt.errorbar([x],[y],xerr=np.array([[x-x1,x2-x]]).T,
                     lw=2,fmt='s',ms=10,capsize=7,capthick=2,
                     color=color,label=func.__name__)

    plt.legend(loc='upper left',fontsize=12)

def test_peak():
    np.random.seed(12345)
    data = generate_distribution()
    peak = ugali.utils.stats.kde_peak(data)
    np.testing.assert_allclose(peak,1.0,atol=1e-2)

def test_median_interval():
    np.random.seed(12345)
    data = generate_distribution()

    peak,[lo,hi] = ugali.utils.stats.median_interval(data,alpha=0.32)
    np.testing.assert_allclose(peak , 1.00, atol=1e-2)
    np.testing.assert_allclose(hi-lo, 2.29, atol=1e-2)

def test_peak_interval():
    np.random.seed(12345)
    data = generate_distribution()

    peak,[lo,hi] = ugali.utils.stats.peak_interval(data,alpha=0.32)
    np.testing.assert_array_less([lo,-hi],[peak,-peak])
    np.testing.assert_allclose(peak , 1.00, atol=1e-2)
    np.testing.assert_allclose(hi-lo, 2.29, atol=1e-2)

def test_min_interval():
    np.random.seed(12345)
    data = generate_distribution()

    # Note that min_interval <= peak_interval
    center,[lo,hi] = ugali.utils.stats.min_interval(data,alpha=0.32)
    np.testing.assert_allclose(center, -0.99, atol=1e-2)
    np.testing.assert_allclose(hi-lo,   2.02, atol=1e-2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
