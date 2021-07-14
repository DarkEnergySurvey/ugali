#!/usr/bin/env python
"""
Tests of ugali stats
"""
import unittest
# Execute tests in order: https://stackoverflow.com/a/22317851/4075339
unittest.TestLoader.sortTestMethodsUsing = None

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

def generate_gaussian():
    p1=[0.0,1.0,int(1e5)]
    p2=[0.0,0.0,0]
    return generate_distribution(p1,p2)

def generate_bimodal():
    p1=[1.0,0.1,10000]
    p2=[-1.0,0.6,30000]
    return generate_distribution(p1,p2)

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


class TestStats(unittest.TestCase):
    """Test the stats """

    def setUp(self):
        np.random.seed(12345)
        self.bimodal = generate_bimodal()
        np.random.seed(12345)
        self.gaussian = generate_gaussian()

    def test_peak(self):
        mean = self.gaussian.mean()
        median = np.median(self.gaussian)
        std  = self.gaussian.std()
        peak = ugali.utils.stats.kde_peak(self.gaussian)

        np.testing.assert_allclose(mean,0.0,atol=1e-2)
        np.testing.assert_allclose(median,0.0,atol=1e-2)
        np.testing.assert_allclose(std ,1.0,atol=1e-2)
        # Only accurate at the ~5% level...
        np.testing.assert_allclose(peak,0.0,atol=1e-1)

        # Bimodal distribution
        peak = ugali.utils.stats.kde_peak(self.bimodal)
        np.testing.assert_allclose(peak,1.0,atol=1e-1)

    def test_mean_interval(self):
        peak,[lo,hi] = ugali.utils.stats.mean_interval(self.gaussian,alpha=0.32)
        np.testing.assert_allclose(peak , 0.00, atol=1e-2)
        np.testing.assert_allclose(hi-lo, 2.00, atol=1e-2)

        peak,[lo,hi] = ugali.utils.stats.mean_interval(self.bimodal,alpha=0.32)
        np.testing.assert_allclose(peak ,-0.50, atol=1e-2)
        np.testing.assert_allclose(hi-lo, 2.01, atol=1e-2)

    def test_median_interval(self):
        peak,[lo,hi] = ugali.utils.stats.median_interval(self.gaussian,alpha=0.32)
        np.testing.assert_allclose(peak , 0.00, atol=1e-2)
        np.testing.assert_allclose(hi-lo, 2.00, atol=1e-2)

        peak,[lo,hi] = ugali.utils.stats.median_interval(self.bimodal,alpha=0.32)
        np.testing.assert_allclose(peak ,-0.75, atol=1e-2)
        np.testing.assert_allclose(hi-lo, 2.44, atol=1e-2)

    def test_peak_interval(self):
        peak,[lo,hi] = ugali.utils.stats.peak_interval(self.gaussian,alpha=0.32)
        np.testing.assert_array_less([lo,-hi],[peak,-peak])
        np.testing.assert_allclose(peak , 0.00, atol=1e-1)
        np.testing.assert_allclose(hi-lo, 1.98, atol=1e-2)

        peak,[lo,hi] = ugali.utils.stats.peak_interval(self.bimodal,alpha=0.32)
        np.testing.assert_array_less([lo,-hi],[peak,-peak])
        np.testing.assert_allclose(peak , 1.00, atol=1e-2)
        np.testing.assert_allclose(hi-lo, 2.29, atol=1e-2)

    def test_min_interval(self):
        center,[lo,hi] = ugali.utils.stats.min_interval(self.gaussian,alpha=0.32)
        np.testing.assert_allclose(center,-0.03, atol=1e-2)
        np.testing.assert_allclose(hi-lo,  1.98, atol=1e-2)

        # Note that min_interval <= peak_interval
        center,[lo,hi] = ugali.utils.stats.min_interval(self.bimodal,alpha=0.32)
        np.testing.assert_allclose(center, -0.99, atol=1e-2)
        np.testing.assert_allclose(hi-lo,   2.02, atol=1e-2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
