#!/usr/bin/env python
import numpy
import numpy as np
import scipy.special

_alpha = 0.32


def interval(best,lo=np.nan,hi=np.nan):
    """
    Pythonized interval for easy output to yaml
    """
    return [float(best),[float(lo),float(hi)]]

def mean_interval(data, alpha=_alpha):
    """
    Interval assuming gaussian posterior.
    """
    mean =np.mean(data)
    sigma = np.std(data)
    scale = scipy.stats.norm.ppf(1-alpha/2.)
    return interval(mean,mean-scale*sigma,mean+scale*sigma)


def median_interval(data, alpha=_alpha):
    """
    Median including bayesian credible interval.
    """
    q = [100*alpha/2., 50, 100*(1-alpha/2.)]
    lo,med,hi = numpy.percentile(data,q)
    return interval(med,lo,hi)
    
def peak(data, bins=100):
    num,edges = np.histogram(data,bins=bins)
    centers = (edges[1:]+edges[:-1])/2.
    return centers[np.argmax(num)]

def kde_peak(data, samples=1000):
    """
    Identify peak using Gaussian kernel density estimator.
    """
    return kde(data,samples)[0]

def kde(data, samples=1000):
    """
    Identify peak using Gaussian kernel density estimator.
    """
    # Clipping of severe outliers to concentrate more KDE samples in the parameter range of interest
    mad = np.median(np.fabs(np.median(data) - data))
    cut = (data > np.median(data) - 5. * mad) & (data < np.median(data) + 5. * mad)
    x = data[cut]
    kde = scipy.stats.gaussian_kde(x)
    # No penalty for using a finer sampling for KDE evaluation except computation time
    values = np.linspace(np.min(x), np.max(x), samples) 
    kde_values = kde.evaluate(values)
    peak = values[np.argmax(kde_values)]
    return values[np.argmax(kde_values)], kde.evaluate(peak)


def peak_interval(data, alpha=_alpha, samples=1000):
    """
    Identify interval using Gaussian kernel density estimator.
    """
    peak = kde_peak(data,samples)
    x = np.sort(data.flat); n = len(x)
    # The number of entries in the interval
    window = int(np.rint((1.0-alpha)*n))
    # The start, stop, and width of all possible intervals
    starts = x[:n-window]; ends = x[window:]
    widths = ends - starts
    # Just the intervals containing the peak
    select = (peak >= starts) & (peak <= ends)
    widths = widths[select]
    if len(widths) == 0:
        raise ValueError('Too few elements for interval calculation')
    min_idx = np.argmin(widths)
    lo = x[min_idx]
    hi = x[min_idx+window]
    return interval(peak,lo,hi)

def min_interval(data, alpha=_alpha):
    x = np.sort(data.flat); n = len(x)
    # The number of entries in the interval
    window = int(np.rint((1.0-alpha)*n))
    # The start, stop, and width of all possible intervals
    starts = x[:n-window]; ends = x[window:]
    widths = ends - starts
    if len(widths) == 0:
        raise ValueError('Too few elements for interval calculation')
    min_idx = np.argmin(widths)
    lo = x[min_idx]
    hi = x[min_idx+window]
    mean = (hi+lo)/2.
    return interval(mean,lo,hi)


def norm_cdf(x):
    # Faster than scipy.stats.norm.cdf
    #https://en.wikipedia.org.wiki/Normal_distribution
    return 0.5*(1 + scipy.special.erf(x/np.sqrt(2)))

def random_pdf(value,pdf,size=None):
    if size is None: size = 1.0
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    fn = scipy.interpolate.interp1d(cdf, range(0, len(cdf)))
    index = np.rint(fn(np.random.uniform(size=size))).astype(int)
    return value[index]

def sky(lon=None,lat=None,size=1):
    """
    Outputs uniform points on sphere from:
    [0 < lon < 360] & [-90 < lat < 90]
    """
    if lon is None:
        umin,umax = 0,1
    else:
        lon = np.asarray(lon)
        lon = np.radians(lon + 360.*(lon<0))
        if   lon.size==1: umin=umax=lon/(2*np.pi)
        elif lon.size==2: umin,umax=lon/(2*np.pi)
        else: raise Exception('...')
        
    if lat is None:
        vmin,vmax = -1,1
    else:
        lat = np.asarray(lat)
        lat = np.radians(90 - lat)
        if   lat.size==1: vmin=vmax=np.cos(lat)
        elif lat.size==2: vmin,vmax=np.cos(lat)
        else: raise Exception('...')

    phi = 2*np.pi*np.random.uniform(umin,umax,size=size)
    theta = np.arcsin(np.random.uniform(vmin,vmax,size=size))
    return np.degrees(phi),np.degrees(theta)

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args

    import pylab as plt
    ax=plt.subplot(221,projection='aitoff')
    ax.grid(True)
    lon,lat = sky(size=1e3)
    lon,lat= np.radians([lon-360.*(lon>180),lat])
    ax.scatter(lon,lat,marker='.',s=2)

    ax=plt.subplot(222,projection='aitoff')
    ax.grid(True)
    lon,lat = sky(size=1e3,lat=[30,45])
    lon,lat= np.radians([lon-360.*(lon>180),lat])
    ax.scatter(lon,lat,marker='.',s=2)

    ax=plt.subplot(223,projection='aitoff')
    ax.grid(True)
    lon,lat = sky(size=1e3,lon=[30,45])
    lon,lat= np.radians([lon-360.*(lon>180),lat])
    ax.scatter(lon,lat,marker='.',s=2)

    ax=plt.subplot(224,projection='aitoff')
    ax.grid(True)
    lon,lat = sky(size=1e3,lon=[0,45],lat=[30,45])
    lon,lat= np.radians([lon-360.*(lon>180),lat])
    ax.scatter(lon,lat,marker='.',s=2)
