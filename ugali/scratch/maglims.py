#!/usr/bin/env python
import numpy as np
import astropy.io.fits as pyfits
import pylab as plt
import glob
import scipy.optimize
from ugali.utils.constants import MAGLIMS 

release = 'y1a1'
maglims = MAGLIMS[release]
#path = "/u/ki/kadrlica/sdss/data/dr10/healpix"
path = "/u/ki/kadrlica/des/data/y1a1/desdm/healpix"
#bands = ['r','g','i']
bands = ['g','r']

# For DR10
#p0 = [-1.0,0.01,0.02]
# For Y1A1
p0 = [-1.0,-1.0, 0.02]

# DR10
#bins = np.linspace(16,24,100)
# Y1A1
bins = np.linspace(17,26,100)

centers = (bins[:-1]+bins[1:])/2.
sample=100
nfiles = 1000
medians = {b:[] for b in bands}

def expf(p,x,b='r'):
    x0 = maglims[b]
    # x0 really not necessary
    return np.exp(p[0]*(x0-x)+p[1])+p[2]

def errf(p,x,y,b='r'):
    ypred = expf(p,x,b)
    return np.fabs(((y-ypred)/ypred)).sum()

def relerr(p,x,y,b='r'):
    ypred = expf(p,x,b)
    return np.fabs(((y-ypred)/ypred)).sum()

def abserr(p,x,y,b='r'):
    ypred = expf(p,x,b)
    return ((y-ypred)**2).sum()


def fitfunc(mag,band='r'):
    # For everything????
    p = [ 0.978 , -2.26,  0.014]
    return expf(p,mag,band)



for filename in sorted(glob.glob(path+'/*'))[:nfiles]:
    print(filename)
    f = pyfits.open(filename)
    for b in bands:
        mag = f[1].data['mag_psf_%s'%b][::sample]
        mag_err = f[1].data['magerr_psf_%s'%b][::sample]
        digi = np.digitize(mag,bins)
        medians[b].append([ np.median(mag_err[digi==i]) for i,c in enumerate(centers)])

    #mag_2 = f[1].data['mag_psf_g'][::100]
    #mag_err_2 = f[1].data['magerr_psf_'][::100]

for b,meds in medians.items():
    fig,ax = plt.subplots()
    plt.title('%s band'%b)
    for med in meds:
        plt.plot(centers,med,color='gray',zorder=9)
    
    x = np.asarray(len(meds)*[centers]).flatten()
    y = np.nan_to_num(np.asarray(meds).flatten())

    relfit = scipy.optimize.minimize(relerr,p0,args=(x,y,b),method='Nelder-Mead')
    absfit = scipy.optimize.minimize(abserr,p0,args=(x,y,b),method='Nelder-Mead')

    plt.plot(centers,expf(relfit.x,centers,b),'-r',lw=2,zorder=10,label='Rel Fit')
    plt.plot(centers,expf(absfit.x,centers,b),'-b',lw=2,zorder=10,label='Abs Fit')

    plt.ylim(0,0.7)
    print(b, 'relfit', relfit.x)
    print(b, 'absfit', absfit.x)
    plt.legend()

plt.ion()
