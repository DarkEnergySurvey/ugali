#!/usr/bin/env python
import numpy
import numpy as np
import scipy.special

def norm_cdf(x):
    # Faster than scipy.stats.norm.cdf
    #https://en.wikipedia.org.wiki/Normal_distribution
    return 0.5*(1 + scipy.special.erf(x/np.sqrt(2)))

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
