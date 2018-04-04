#!/usr/bin/env python
import yaml 
import numpy as np
from os.path import join

import matplotlib, os
try:             os.environ['DISPLAY']
except KeyError: matplotlib.use('Agg')
from matplotlib import font_manager

import pylab as plt

from ugali.utils.shell import mkdir
import ugali.analysis.loglike
from ugali.utils.projector import cel2gal, gal2cel
import ugali.utils.plotting
from ugali.utils.config import Config
from ugali.analysis.kernel import Disk
from ugali.isochrone import Padova
import ugali.analysis.source

from dsphs.like.lnlfn import ProfileLimit

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from matplotlib import patches 
import mpl_toolkits.axes_grid1.axes_divider as axes_divider
from collections import OrderedDict as odict

#def scan(loglike,xdict,ydict):
#    xpar,xvals = xdict.items()[0]
#    ypar,yvals = ydict.items()[0]
#    nx,ny = len(xvals),len(yvals)
# 
#    lnl,rich = [],[]
#    for i in range(ny):
#        print i,yvals[i]
#        loglike.value(**{ypar:yvals[i]})
#        for j in range(nx):
#            loglike.value(**{xpar:xvals[j]})
#            l,r,junk = loglike.fit_richness()
#            rich.append(r)
#            lnl.append(l)
#    return np.array(lnl).reshape(ny,nx),np.array(rich).reshape(ny,nx)


def scan(loglike,xdict,ydict,zdict):
    xpar,xvals = list(xdict.items())[0]
    ypar,yvals = list(ydict.items())[0]
    zpar,zvals = list(zdict.items())[0]

    nx,ny,nz = len(xvals),len(yvals),len(zvals)
    
    val,lnl,rich = [],[],[]
    for i in range(nz):
        print(i,"%s: %.2f"%(zpar,zvals[i]))
        loglike.set_params(**{zpar:zvals[i]})
        for j in range(ny):
            loglike.set_params(**{ypar:yvals[j]})
            for k in range(nx):
                loglike.set_params(**{xpar:xvals[k]})
                l,r,junk = loglike.fit_richness()
                rich.append(r)
                lnl.append(l)
                val.append((zvals[i],yvals[j],xvals[k]))
    return np.array(val),np.array(lnl).reshape(nz,ny,nx),np.array(rich).reshape(nz,ny,nx)

def plot(lnl):
    pass

bounds = odict([
        ('richness',lambda r: [r-np.sqrt(r),r+np.sqrt(r)]),
        #('lon',lambda l: [l-0.03,l+0.03]),
        #('lat',lambda b: [b-.03,b+0.03]),
        ('lon',lambda l: [l-0.05,l+0.05]),
        ('lat',lambda b: [b-0.05,b+0.05]),
        ('extension',lambda e: [0.1*e,10.*e]),
        ('ellipticity',lambda e: [0.0001,0.8]),
        ('position_angle',lambda e: [0,180]),
        ('distance_modulus',lambda m: [m-1,m+1]),
        ('age',lambda a: [8,13.5]),
        ('metallicity',lambda z: [1e-4,1e-3]),
        ])


if __name__ == "__main__":
    from ugali.utils.parser import Parser
    parser = Parser(description="Plot fit diagnostics")
    parser.add_coords(radius=True,targets=True)
    parser.add_config(default='config_y2q1_mcmc.yaml',nargs='?')
    parser.add_force()
    parser.add_argument('-n','--name',default=None)
    parser.add_argument('--xpar',default='extension',help="Fast parameter")
    parser.add_argument('--xbins',default=10)
    parser.add_argument('--ypar',default='distance_modulus',help="Slow parameter")
    parser.add_argument('--ybins',default=10)
    parser.add_argument('--zpar',default='age',help="Slowest parameter")
    parser.add_argument('--zbins',default=10)
    parser.add_argument('--alpha',default=0.1)

    opts = parser.parse_args()
    alpha = opts.alpha

    config = opts.config
    dirname = 'mcmc_v01'
    srcmdl = 'srcmdl.yaml'

    if opts.name: names = [opts.name]
    else: names = opts.names
    outdir = mkdir('plots')

    a = 13.5
    z = 0.0001
    for name in names:
        if opts.name is not None:
            if name.lower() != opts.name.lower(): continue
        print(name)
        #ra,dec = params['ra'],params['dec']
        #lon,lat = cel2gal(ra,dec)
        #params['lon'],params['lat'] = lon,lat
        #params['age'] = a
        #params['metallicity'] = z

        #srcmdl = join(dirname,'%s_mcmc.yaml'%name)

        source = ugali.analysis.source.Source()
        source.load(srcmdl,name)
        loglike = ugali.analysis.loglike.createLoglike(config,source)
        params = source.params

        xpar = opts.xpar
        ypar = opts.ypar
        zpar = opts.zpar
        xval = params[xpar]
        yval = params[ypar]
        zval = params[zpar]

        fmt = '%s = %.5g [+%.2g,-%.2g]'

        #loglike = ugali.analysis.loglike.createLoglike(config,lon,lat)
        ##loglike.models['color'] = Padova(age=12.5,z=0.0002,hb_spread=0)
        #loglike.value(**params)

        for p in [xpar,ypar]:
            b = bounds[p]
            v = source.params[p].value
            source.params[p].set_bounds(b(v))

        xmin,xmax = source.params[xpar].bounds
        ymin,ymax = source.params[ypar].bounds
        zmin,zmax = source.params[zpar].bounds

        x = np.linspace(xmin,xmax,opts.xbins)
        y = np.linspace(ymin,ymax,opts.ybins)
        z = np.linspace(zmin,zmax,opts.zbins)
        nx,ny,nz = len(x),len(y),len(z)

        val,lnl,rich =  scan(loglike,{xpar:x},{ypar:y},{zpar:z})
        #lnl,richness = scan(loglike,{xpar:x},{ypar:y},{zpar:z})
        ts = 2*lnl

        xx,yy,zz = np.meshgrid(x,y,z)

        maxlnl = np.max(lnl)
        idx =np.argmax(lnl)
        zidx,yidx,xidx = np.unravel_index(idx,lnl.shape)
        print(list(zip([xpar,ypar,zpar],[x[xidx],y[yidx],z[zidx]])))

        # Probably a better way to do the profile with more variables...
        #stackoverflow.com/q/30589211
        #lnlike = np.max(lnl.shape(nz,-1),axis=1)

        results =dict()
        for i,(p,v) in enumerate(zip([xpar,ypar,zpar],[x,y,z])):
            # Not great, but clear
            lnlike = np.max(np.max(np.swapaxes(lnl,i,0),axis=0),axis=0)
            lnlfn = ProfileLimit(v,lnlike)
            lnlfn._mle = v[np.argmax(lnlike)]
            lnlfn._fmax = np.max(lnlike)
            mle = lnlfn._mle
            try:
                lo = lnlfn.getLowerLimit(alpha/2)
            except ValueError:
                lo = np.nan
            try: 
                hi = lnlfn.getUpperLimit(alpha/2)
            except ValueError:
                hi = np.nan
            results[p] = [lnlfn, mle, [lo,hi]]

        ##richness = np.array(richness).reshape(xx.shape)
        #like = np.exp(lnl-lnl.max())
        #maxlike = np.max(like)
        #idx =np.argmax(like)
        #yidx,xidx = np.unravel_index(idx,lnl.shape)
        #print x[xidx],y[yidx]

        #loglike.value(**{xpar:x[xidx],ypar:y[yidx]})
        #richs = np.logspace(np.log10(richness.flat[idx])-1,np.log10(richness.flat[idx])+1,100)
        #rich_lnlfn = ProfileLimit(richs,np.array([loglike.value(richness=r) for r in richs]))
         
        """
        # Plotting... lot's of plotting
        lnlmax = np.max(lnl)
        data = lnl - lnlmax
        im_kwargs = dict(extent = [xmin,xmax,ymin,ymax],aspect='auto',
                         interpolation='none',origin='lower')
        mle_kwargs = dict(color='black',ls='-')
        err_kwargs = dict(color='black',ls='--')
        cs_kwargs = dict(colors='0.66',ls='-',**im_kwargs)
        fig,ax = plt.subplots()
        ax.imshow(data,**im_kwargs)

        levels = odict([
                (-1.0/2. ,'68%'),
                (-1.64/2.,'80%'),
                (-2.71/2.,'90%'),
                (-3.84/2.,'95%'),
                ])
        cs = plt.contour(data,levels=levels.keys(),**cs_kwargs)
        plt.clabel(cs,fmt=levels,inline=1,fontsize=8)
                    
        plt.plot(x[xidx],y[yidx],'x',ms=10,mew=5,**mle_kwargs)
        #plt.plot(results[0][0],results[1][0],'bx',ms=10,mew=5)
        plt.plot(x,y[np.argmax(like,axis=0)],lw=1.5,**err_kwargs)
        
        #ax2 = ugali.utils.plotting.draw_sum_slices(data)
        ax2 = ugali.utils.plotting.draw_max_slices(data)
        
        ann_kwargs = dict(xycoords='axes fraction',fontsize=8,
                          bbox={'facecolor':'w'})
            
        for i, r in enumerate(results):
            attr = 'axvline' if i==0 else 'axhline'
            par = xpar if i==0 else ypar
            getattr(ax,attr)(r[0],**mle_kwargs)
            getattr(ax2[i],attr)(r[0],**mle_kwargs)
            for l in r[1]:
                getattr(ax,attr)(l,**err_kwargs)
                getattr(ax2[i],attr)(l,**err_kwargs)
            ax.annotate(fmt%(par,r[0],r[1][1]-r[0],r[0]-r[1][0]),
                        xy=(0.05,.9-0.05*i),**ann_kwargs)

        ax.annotate('TS = %.0f'%(2*lnlmax),
                    xy=(0.75,.9),**ann_kwargs)

        ax.set_xlabel(xpar)
        ax.set_ylabel(ypar)

        outfile = os.path.join(outdir,'%s_%s_%s.png'%(name.lower(),xpar,ypar))
        plt.savefig(outfile,bbox_inches='tight')
        """
    plt.ion()

    """
        rich_mle = rich_lnlfn._mle
        rich_lo = rich_lnlfn.getLowerLimit(alpha/2)
        rich_hi = rich_lnlfn.getUpperLimit(alpha/2)

        fig,axes = plt.subplots(1,3,figsize=(12,4))
        log_exts = np.log10(exts)
         
        levels = [ts.max(),ts.max()-2.71/2, ts.max() - 3.83/2]

        ax = axes[2]
        im = ax.imshow(ts,**kwargs)
        #draw_slices(ts)
        #ax.set_xlim(extent[0],extent[1])
        #ax.set_ylim(extent[2],extent[3])
        cb = plt.colorbar(im)
        cb.set_label('TS')
         
        plt.contour(ts,**kwargs)
        plt.plot(np.log10(xx).flat[np.argmax(ts)],yy.flat[np.argmax(ts)],'kx',ms=10,mew=5)
        ann_kwargs = dict(xycoords='axes fraction',fontsize=8,bbox={'facecolor':'w'})

        outfile = os.path.join(lnldir,'%s_ext_mod_tsmap.png'%name.lower())
        plt.savefig(outfile,bbox_inches='tight')
        
        print 'Max TS:',ts.max()
        print 'Richness MLE:',rich_mle,[rich_lo,rich_hi]
        print 'Extension MLE:',ext_mle,[ext_lo,ext_hi]
        print 'Distance Modulus MLE:', mod_mle,[mod_lo,mod_hi]
    """
