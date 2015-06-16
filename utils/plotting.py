"""
Basic plotting tools.
"""
import os
import collections

import matplotlib
try:             os.environ['DISPLAY']
except KeyError: matplotlib.use('Agg')

import numpy
import numpy as np
import pylab
import pylab as plt
import healpy
import pyfits

from mpl_toolkits.axes_grid1 import AxesGrid    
from matplotlib.ticker import MaxNLocator

import ugali.utils.config
import ugali.observation.roi
import ugali.observation.catalog
import ugali.utils.skymap
import ugali.utils.projector
import ugali.utils.healpix
import ugali.analysis.isochrone2

from ugali.utils.healpix import ang2pix
from ugali.utils.projector import mod2dist,gal2cel,cel2gal

from ugali.utils.logger import logger
#pylab.ion()

############################################################

def histogram(title, title_x, title_y,
              x, bins_x):
    """
    Plot a basic histogram.
    """
    pylab.figure()
    pylab.hist(x, bins_x)
    pylab.xlabel(title_x)
    pylab.ylabel(title_y)
    pylab.title(title)

############################################################

def twoDimensionalHistogram(title, title_x, title_y,
                            z, bins_x, bins_y,
                            lim_x=None, lim_y=None,
                            vmin=None, vmax=None):
    """
    Create a two-dimension histogram plot or binned map.

    If using the outputs of numpy.histogram2d, remember to transpose the histogram.

    INPUTS
    """
    pylab.figure()

    mesh_x, mesh_y = numpy.meshgrid(bins_x, bins_y)

    if vmin != None and vmin == vmax:
        pylab.pcolor(mesh_x, mesh_y, z)
    else:
        pylab.pcolor(mesh_x, mesh_y, z, vmin=vmin, vmax=vmax)
    pylab.xlabel(title_x)
    pylab.ylabel(title_y)
    pylab.title(title)
    pylab.colorbar()

    if lim_x:
        pylab.xlim(lim_x[0], lim_x[1])
    if lim_y:
        pylab.ylim(lim_y[0], lim_y[1])
        
############################################################

def twoDimensionalScatter(title, title_x, title_y,
                          x, y,
                          lim_x = None, lim_y = None,
                          color = 'b', size = 20, alpha=None):
    """
    Create a two-dimensional scatter plot.

    INPUTS
    """
    pylab.figure()

    pylab.scatter(x, y, c=color, s=size, alpha=alpha, edgecolors='none')
    
    pylab.xlabel(title_x)
    pylab.ylabel(title_y)
    pylab.title(title)
    if type(color) is not str:
        pylab.colorbar()

    if lim_x:
        pylab.xlim(lim_x[0], lim_x[1])
    if lim_y:
        pylab.ylim(lim_y[0], lim_y[1])

############################################################

def zoomedHealpixMap(title, map, lon, lat, radius,
                     xsize=1000, **kwargs):
    """
    Inputs: lon (deg), lat (deg), radius (deg)
    """
    reso = 60. * 2. * radius / xsize # Deg to arcmin
    healpy.gnomview(map=map, rot=[lon, lat, 0], title=title, xsize=xsize, reso=reso, degree=False, **kwargs)

############################################################

def projScatter(lon, lat, **kwargs):
    """
    Create a scatter plot on HEALPix projected axes.
    Inputs: lon (deg), lat (deg)
    """
    healpy.projscatter(lon, lat, lonlat=True, **kwargs)

############################################################


def sparseHealpixFiles(title, infiles, field='MAGLIM',**kwargs):
                       
    """
    Inputs: field
    """
    map = ugali.utils.skymap.readSparseHealpixMaps(infiles,field)
    ax = healpy.mollview(map=map, title=title, **kwargs)
    return ax, map
    
############################################################

def getSDSSImage(ra,dec,radius=1.0,xsize=800,opt='GMPL'):
    """
    http://skyserver.sdss3.org/dr9/en/tools/chart/chart.asp

    radius (degrees)
    opts: (G) Grid, (L) Label, P (PhotoObj), S (SpecObj), O (Outline), (B) Bounding Box, 
          (F) Fields, (M) Mask, (Q) Plates, (I) Invert
    """
    import subprocess
    import tempfile

    url="http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?"
    scale = 2. * radius * 3600. / xsize
    params=dict(ra=ra,dec=dec,
                width=xsize,height=xsize,
                scale=scale,opt=opt)
    query='&'.join("%s=%s"%(k,v) for k,v in params.items())
    
    tmp = tempfile.NamedTemporaryFile(suffix='.jpeg')
    cmd='wget -O %s "%s"'%(tmp.name,url+query)
    subprocess.call(cmd,shell=True)
    im = pylab.imread(tmp.name)
    tmp.close()
    return im

############################################################

class BasePlotter(object):
    def __init__(self,glon,glat,config,radius=1.0):
        self.glon,self.glat = glon,glat
        self.ra,self.dec = ugali.utils.projector.galToCel(self.glon,self.glat)
        self.config = ugali.utils.config.Config(config)
        self.roi = ugali.observation.roi.ROI(self.config,self.glon,self.glat)
        self.nside = self.config.params['coords']['nside_pixel']
        self.radius = radius
        xsize=800
        reso = 60. * 2. * radius / xsize
        self.image_kwargs = dict(ra=self.ra,dec=self.dec,radius=self.radius,xsize=xsize,opt='GML')
        self.gnom_kwargs = dict(rot=[self.ra,self.dec],reso=reso,xsize=xsize,coord='GC',return_projected_map=True)
        self.label_kwargs = dict(xy=(0.05,0.05),xycoords='axes fraction', xytext=(0, 0), 
                                 textcoords='offset points',ha='left', va='bottom',size=10,
                                 bbox={'boxstyle':"round",'fc':'1'}, zorder=10)
        
    def _create_catalog(self):
        if hasattr(self,'catalog'): return
        catalog = ugali.observation.catalog.Catalog(self.config,roi=self.roi)
        sep = ugali.utils.projector.angsep(self.glon, self.glat, catalog.lon, catalog.lat)
        radius = self.radius*np.sqrt(2)
        cut = (sep < radius)
        self.catalog = catalog.applyCut(cut)

    def drawROI(self, ax, value=None, pixel=None):
        roi_map = numpy.array(healpy.UNSEEN * np.ones(healpy.nside2npix(self.nside)))
        
        if value is None:
            #roi_map[self.pixels] = ugali.utils.projector.angsep(self.lon, self.lat, self.centers_lon, self.centers_lat)
            roi_map[self.roi.pixels] = 1
            roi_map[self.roi.pixels_annulus] = 0
            roi_map[self.roi.pixels_target] = 2
        elif value is not None and pixel is None:
            roi_map[self.pixels] = value
        elif value is not None and pixel is not None:
            roi_map[pixel] = value
        else:
            print 'ERROR: count not parse input'
        im = healpy.gnomview(roi_map,**self.gnom_kwargs)

    def drawImage(self,ax,invert=True):
        if self.config['data']['survey']=='sdss':
            # Optical Image
            im = ugali.utils.plotting.getSDSSImage(**self.image_kwargs)
            # Flipping JPEG:
            # https://github.com/matplotlib/matplotlib/issues/101
            if invert: im =ax.imshow(im[::-1])
            else:      im =ax.imshow(im)
            try:
                ax.cax.axis["right"].toggle(ticks=False, ticklabels=False)
                ax.cax.axis["top"].toggle(ticks=False, ticklabels=False)
            except AttributeError: 
                pass
        ax.annotate("Image",**self.label_kwargs)
        return ax

    def drawStellarDensity(self,ax):
        # Stellar Catalog
        self._create_catalog()
        catalog = self.catalog
        #catalog=ugali.observation.catalog.Catalog(self.config,roi=self.roi)
        pix = ang2pix(self.nside, catalog.lon, catalog.lat)
        counts = collections.Counter(pix)
        pixels, number = numpy.array(sorted(counts.items())).T
        star_map = healpy.UNSEEN * numpy.ones(healpy.nside2npix(self.nside))
        star_map[pixels] = number
        star_map = numpy.where(star_map == 0, healpy.UNSEEN, star_map)
    
        im = healpy.gnomview(star_map,**self.gnom_kwargs)
        healpy.graticule(dpar=1,dmer=1,color='0.5',verbose=False)
        pylab.close()
        im = ax.imshow(im,origin='bottom')
        try: ax.cax.colorbar(im)
        except: pylab.colorbar(im)
        ax.annotate("Stars",**self.label_kwargs)

    def drawMask(self,ax, mask=None):
        # MAGLIM Mask
        if mask is None:
            filenames = self.config.getFilenames()
            catalog_pixels = self.roi.getCatalogPixels()
            mask_map = ugali.utils.skymap.readSparseHealpixMaps(filenames['mask_1'][catalog_pixels], field='MAGLIM')
        else:
            mask_map = healpy.UNSEEN*np.ones(healpy.nside2npix(self.config['coords']['nside_pixel']))
            mask_map[mask.roi.pixels] = mask.mask_1.mask_roi_sparse
        mask_map = numpy.where(mask_map == healpy.UNSEEN, 0, mask_map)
         
        im = healpy.gnomview(mask_map,**self.gnom_kwargs)
        healpy.graticule(dpar=1,dmer=1,color='0.5',verbose=False)
        pylab.close()
        im = ax.imshow(im,origin='bottom')
        try: ax.cax.colorbar(im)
        except: pylab.colorbar(im)
        ax.annotate("Mask",**self.label_kwargs)

    def drawTS(self,ax, filename=None, zidx=0):
        if not filename:
            #dirname = self.config.params['output2']['searchdir']
            #basename = self.config.params['output2']['mergefile']
            #filename = os.path.join(dirname,basename)
            filename = self.config.mergefile

        results=pyfits.open(filename)[1]

        pixels,values = results.data['pix'],2*results.data['log_likelihood']
        if values.ndim == 1: values = values.reshape(-1,1)
        ts_map = healpy.UNSEEN * numpy.ones(healpy.nside2npix(self.nside))
        # Sum through all distance_moduli
        #ts_map[pixels] = values.sum(axis=1)
        # Just at maximum slice from object

        ts_map[pixels] = values[:,zidx]

        im = healpy.gnomview(ts_map,**self.gnom_kwargs)
        healpy.graticule(dpar=1,dmer=1,color='0.5',verbose=False)
        pylab.close()
        im = ax.imshow(im,origin='bottom')
        try: ax.cax.colorbar(im)
        except: pylab.colorbar(im)
        ax.annotate("TS",**self.label_kwargs)

    def drawCatalog(self, ax):
        # Stellar Catalog
        self._create_catalog()
        healpy.projscatter(self.catalog.lon,self.catalog.lat,c='k',marker='.',lonlat=True,coord=self.gnom_kwargs['coord'])
        ax.annotate("Stars",**self.label_kwargs)

    def drawSpatial(self, ax):
        # Stellar Catalog
        self._create_catalog()
        cut = (self.catalog.color > 0) & (self.catalog.color < 1)
        catalog = self.catalog.applyCut(cut)
        ax.scatter(catalog.lon,catalog.lat,c='k',marker='.',s=1)
        ax.set_xlim(self.glon-0.5,self.glon+0.5)
        ax.set_ylim(self.glat-0.5,self.glat+0.5)
        ax.set_xlabel('GLON (deg)')
        ax.set_ylabel('GLAT (deg)')
        #ax.annotate("Stars",**self.label_kwargs)

    def drawCMD(self, ax, radius=None, zidx=None):
        import ugali.analysis.isochrone2

        if zidx is not None:
            #dirname = self.config.params['output2']['searchdir']
            #basename = self.config.params['output2']['mergefile']
            #filename = os.path.join(dirname,basename)
            filename = self.config.mergefile
            logger.debug("Opening %s..."%filename)
            f = pyfits.open(filename)
            distance_modulus = f[2].data['DISTANCE_MODULUS'][zidx]

            iso = ugali.analysis.isochrone2.Padova(age=12,z=0.0002,mod=distance_modulus)
            drawIsochrone(iso,ls='',marker='.',ms=1,c='k')

            #for ii, name in enumerate(self.config.params['isochrone']['infiles']):
            #    print ii, name
            #    drawIsochrone(ax, self.config, distance_modulus,lw=25,c='0.5')
            #    drawIsochrone(ax, self.config, distance_modulus,ls='',marker='.',ms=1,c='k')
            #    #isochrone = ugali.analysis.isochrone.Isochrone(self.config, name)
            #    #mag = isochrone.mag + distance_modulus
            #    #ax.scatter(isochrone.color,mag, color='0.5', s=750, zorder=0)
        
        # Stellar Catalog
        self._create_catalog()
        if radius is not None:
            sep = ugali.utils.projector.angsep(self.glon,self.glat,self.catalog.lon,self.catalog.lat)
            cut = (sep < radius)
            catalog_cmd = self.catalog.applyCut(cut)
        else:
            catalog_cmd = self.catalog
    
        ax.scatter(catalog_cmd.color, catalog_cmd.mag,color='b',marker='.',s=1)
        ax.set_xlim(self.roi.bins_color[0],self.roi.bins_color[-1])
        ax.set_ylim(self.roi.bins_mag[-1],self.roi.bins_mag[0])
        ax.set_xlabel('Color (mag)')
        ax.set_ylabel('Magnitude (mag)')

        try:    ax.cax.colorbar(im)
        except: pass
        ax.annotate("Stars",**self.label_kwargs)


    def drawMembership(self, ax, radius=None, zidx=0, mc_source_id=1):
        import ugali.analysis.scan

        #dirname = self.config.params['output2']['searchdir']
        #basename = self.config.params['output2']['mergefile']
        #filename = os.path.join(dirname,basename)
        filename = self.config.mergefile
        logger.debug("Opening %s..."%filename)
        f = pyfits.open(filename)
        distance_modulus = f[2].data['DISTANCE_MODULUS'][zidx]

        for ii, name in enumerate(self.config.params['isochrone']['infiles']):
            print ii, name
            isochrone = ugali.analysis.isochrone.Isochrone(self.config, name)
            mag = isochrone.mag + distance_modulus
            ax.scatter(isochrone.color,mag, color='0.5', s=800, zorder=0)
            #ax.plot(isochrone.color,mag, lw=20, color='0.5', zorder=0)


        pix = ang2pix(self.nside, self.glon, self.glat)
        likelihood_pix = ugali.utils.skymap.superpixel(pix,self.nside,self.config.params['coords']['nside_likelihood'])
        config = self.config
        scan = ugali.analysis.scan.Scan(self.config,likelihood_pix)
        likelihood = scan.likelihood
        distance_modulus_array = [self.config.params['scan']['distance_modulus_array'][zidx]]
        likelihood.precomputeGridSearch(distance_modulus_array)
        likelihood.gridSearch()
        p = likelihood.membershipGridSearch()

        sep = ugali.utils.projector.angsep(self.glon, self.glat, likelihood.catalog.lon, likelihood.catalog.lat)
        radius = self.radius if radius is None else radius
        cut = (sep < radius)
        catalog = likelihood.catalog.applyCut(cut)
        p = p[cut]

        cut_mc_source_id = (catalog.mc_source_id == mc_source_id)
        ax.scatter(catalog.color[cut_mc_source_id], catalog.mag[cut_mc_source_id], c='gray', s=100, edgecolors='none')
        sc = ax.scatter(catalog.color, catalog.mag, c=p, edgecolors='none')

        ax.set_xlim(likelihood.roi.bins_color[0], likelihood.roi.bins_color[-1])
        ax.set_ylim(likelihood.roi.bins_mag[-1], likelihood.roi.bins_mag[0])
        ax.set_xlabel('Color (mag)')
        ax.set_ylabel('Magnitude (mag)')
        try: ax.cax.colorbar(sc)
        except: pylab.colorbar(sc)

    def plotDistance(self):
        filename = self.config.mergefile
        logger.debug("Opening %s..."%filename)
        f = pyfits.open(filename)
        pixels,values = f[1].data['pix'],2*f[1].data['LOG_LIKELIHOOD']
        if values.ndim == 1: values = values.reshape(-1,1)
        distances = f[2].data['DISTANCE_MODULUS']
        if distances.ndim == 1: distances = distances.reshape(-1,1)
        #pixels,values = f[1].data['pix'],f[1].data['fraction_observable']
        #pixels,values = f[1].data['pix'],f[1].data['richness']*f[1].data['fraction_observable']
        ts_map = healpy.UNSEEN * numpy.ones(healpy.nside2npix(self.nside))

        ndim = len(distances)
        nrows = int(numpy.sqrt(ndim))
        ncols = ndim // nrows + (ndim%nrows > 0)

        fig = pylab.figure()
        axes  = AxesGrid(fig, 111, nrows_ncols = (nrows, ncols),axes_pad=0,
                         label_mode='1', cbar_mode='single',cbar_pad=0,cbar_size='5%',
                         share_all=True,add_all=False)

        images = []
        for i,val in enumerate(values.T):
            ts_map[pixels] = val
            
            im = healpy.gnomview(ts_map,**self.gnom_kwargs)
            pylab.close()
            images.append(im)
        data = numpy.array(images); mask = (data == healpy.UNSEEN)
        images = numpy.ma.array(data=data,mask=mask)
        vmin = numpy.ma.min(images)
        vmax = numpy.ma.max(images)

        for i,val in enumerate(values.T):
            ax = axes[i]
            im = ax.imshow(images[i],origin='bottom',vmin=vmin,vmax=vmax)
            ax.cax.colorbar(im)
            
            #ax.annotate(r"$\mu = %g$"%distances[i],**self.label_kwargs)
            ax.annotate(r"$d = %.0f$ kpc"%mod2dist(distances[i]),**self.label_kwargs)
            ax.axis["left"].major_ticklabels.set_visible(False) 
            ax.axis["bottom"].major_ticklabels.set_visible(False) 
            fig.add_axes(ax)
            fig.add_axes(ax.cax)
        return fig,axes


    def plot3(self):
        fig = pylab.figure(figsize=(8,4))
        axes = AxesGrid(fig, 111,nrows_ncols = (1, 3),axes_pad=0.1,
                        cbar_mode='each',cbar_pad=0,cbar_size='5%',
                        cbar_location='top',share_all=True)
        for ax in axes:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        self.drawImage(axes[0])
        self.drawTS(axes[1])
        #self.drawStellarDensity(axes[1])
        self.drawMask(axes[2])
        return fig,axes


    def plot4(self):
        fig = pylab.figure()
        axes = AxesGrid(fig, 111,nrows_ncols = (2, 2),axes_pad=0.25,
                        cbar_mode='each',cbar_pad=0,cbar_size='5%',
                        share_all=True)
        for ax in axes:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        self.drawImage(axes[0])
        self.drawStellarDensity(axes[1])
        self.drawMask(axes[2])
        self.drawTS(axes[3])
        return fig,axes

    plot = plot3


class ObjectPlotter(BasePlotter):
    """ For plotting 'Objects' identified through candidate search. """

    def __init__(self,obj,config,radius=1.0):
        self.obj = obj
        glon,glat = self.obj['GLON'],self.obj['GLAT']

        self.zidx = self.obj['ZIDX_MAX'] 
        super(ObjectPlotter,self).__init__(glon,glat,config,radius)

    def drawTS(self,ax, filename=None, zidx=None):
        if zidx is None: zidx = self.zidx
        super(ObjectPlotter,self).drawTS(ax,filename,zidx)

    def drawCMD(self,ax, radius=None, zidx=None):
        if zidx is None: zidx = self.zidx
        super(ObjectPlotter,self).drawCMD(ax,radius,zidx)

    def drawMembership(self, ax, radius=None, zidx=None, mc_source_id=1):
        if zidx is None: zidx = self.zidx
        super(ObjectPlotter,self).drawMembership(ax,radius,zidx,mc_source_id)


def plot_candidates(candidates, config, ts_min=50, outdir='./'):
    
    for candidate in candidates:
        if candidate['TS'] < ts_min: continue
        logger.info("Plotting %s (%.2f,%.2f)..."%(candidate['name'],candidate['glon'],candidate['glat']))
        plotter = ugali.utils.plotting.ObjectPlotter(candidate,config)
        fig,ax = plotter.plot4()
        basename = '%s_plot.png'%candidate['name']
        outfile = os.path.join(outdir,basename)
        plt.savefig(outfile)


###################################################


def draw_slices(ax, hist, **kwargs):
    """ Draw horizontal and vertical slices through histogram """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    kwargs.setdefault('ls','-')

    data = hist
    npix = np.array(data.shape)
    xlim,ylim = plt.array(zip([0,0],npix-1))

    # Slices
    vslice = data.sum(axis=0)
    hslice = data.sum(axis=1)
    # Bin centers
    xbin = np.linspace(xlim[0],xlim[1],len(vslice))+0.5 
    ybin = np.linspace(ylim[0],ylim[1],len(hslice))+0.5
    divider = make_axes_locatable(ax)

    #gh2 = pywcsgrid2.GridHelperSimple(wcs=self.header, axis_nums=[2, 1])
    hax = divider.append_axes("right", size=1.2, pad=0.05,sharey=ax)
                              #axes_class=axes_divider.LocatableAxes)
    hax.axis["left"].toggle(label=False, ticklabels=False)
    #hax.plot(hslice, plt.arange(*ylim)+0.5,'-') # Bin center
    hax.plot(hslice, ybin, **kwargs) # Bin center
    hax.xaxis.set_major_locator(MaxNLocator(4,prune='both'))

    #gh1 = pywcsgrid2.GridHelperSimple(wcs=self.header, axis_nums=[0, 2])
    vax = divider.append_axes("top", size=1.2, pad=0.05, sharex=ax)
                              #axes_class=axes_divider.LocatableAxes)
    vax.axis["bottom"].toggle(label=False, ticklabels=False)
    vax.plot(xbin, vslice, **kwargs) 
    vax.xaxis.set_major_locator(MaxNLocator(4,prune='lower'))

    return hax, vax

def plotKernel(kernel):
    fig = plt.figure()
    axes = AxesGrid(fig, 111, nrows_ncols = (1,1),
                    cbar_mode='none',cbar_pad=0,cbar_size='5%',
                    cbar_location='top', share_all=True)
    drawKernel(axes[0],kernel)

def drawKernelHist(ax, kernel):
    ext = kernel.extension
    theta = kernel.theta
    lon, lat = kernel.lon, kernel.lat
    xmin,xmax = -5*ext,5*ext
    ymin,ymax = -5*ext,5*ext,
    x = np.linspace(xmin,xmax,100)+kernel.lon
    y = np.linspace(ymin,ymax,100)+kernel.lat

    xx,yy = np.meshgrid(x,y)
    zz = kernel.pdf(xx,yy)
    im = ax.imshow(zz)#,extent=[xmin,xmax,ymin,ymax])
    hax,vax = draw_slices(ax,zz,color='k')

    mc_lon,mc_lat = kernel.sample(1e5)
    hist,xedges,yedges = np.histogram2d(mc_lon,mc_lat,bins=[len(x),len(y)],
                                        range=[[x.min(),x.max()],[y.min(),y.max()]])
    xbins,ybins = np.arange(hist.shape[0])+0.5,np.arange(hist.shape[1])+0.5

    vzz = zz.sum(axis=0)
    hzz = zz.sum(axis=1)
    vmc = hist.sum(axis=0)
    hmc = hist.sum(axis=1)

    vscale = vzz.max()/vmc.max()
    hscale = hzz.max()/hmc.max()

    kwargs = dict(marker='.',ls='',color='r')
    hax.errorbar(hmc*hscale, ybins, xerr=np.sqrt(hmc)*hscale,**kwargs)
    vax.errorbar(xbins, vmc*vscale,yerr=np.sqrt(vmc)*vscale,**kwargs) 

    ax.set_ylim(0,len(y))
    ax.set_xlim(0,len(x))

    #try: ax.cax.colorbar(im)
    #except: pylab.colorbar(im)

    #a0 = np.array([0.,0.])
    #a1 =kernel.a*np.array([np.sin(np.deg2rad(theta)),-np.cos(np.deg2rad(theta))])
    #ax.plot([a0[0],a1[0]],[a0[1],a1[1]],'-ob')
    # 
    #b0 = np.array([0.,0.])
    #b1 =kernel.b*np.array([np.cos(np.radians(theta)),np.sin(np.radians(theta))])
    #ax.plot([b0[0],b1[0]],[b0[1],b1[1]],'-or')    

    label_kwargs = dict(xy=(0.05,0.05),xycoords='axes fraction', xytext=(0, 0), 
                        textcoords='offset points',ha='left', va='bottom',size=10,
                        bbox={'boxstyle':"round",'fc':'1'}, zorder=10)
    norm = zz.sum() * (x[1]-x[0])**2
    ax.annotate("Sum = %.2f"%norm,**label_kwargs)
        
    #ax.set_xlabel(r'$\Delta$ LON (deg)')
    #ax.set_ylabel(r'$\Delta$ LAT (deg)')

###################################################

def plotMembership(config, data=None, kernel=None, isochrone=None,**kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    config = ugali.utils.config.Config(config)
    if isinstance(data,basestring):
        data = pyfits.open(data)[1].data

    kwargs.setdefault('s',20)
    kwargs.setdefault('edgecolor','none')
    kwargs.setdefault('vmin',0)
    kwargs.setdefault('vmax',1)
    kwargs.setdefault('zorder',3)

    try: 
        sort = np.argsort(data['PROB'])
        prob = data['PROB'][sort]
    except:
        prob = np.zeros(len(data['RA']))+1

    lon,lat = data['RA'][sort],data['DEC'][sort]
    color = data['COLOR'][sort]
    cut = (prob > 0)

    # ADW: Sometimes may be mag_2
    mag = data[config['catalog']['mag_1_field']][sort]
    mag_err_1 = data[config['catalog']['mag_err_1_field']][sort]
    mag_err_2 = data[config['catalog']['mag_err_2_field']][sort]

    fig,axes = plt.subplots(1,2,figsize=(10,5))
        
    #proj = ugali.utils.projector.Projector(np.median(lon),np.median(lat))
    #x,y = proj.sphereToImage(lon,lat)
    #sc = axes[0].scatter(x,y,c=prob,vmin=0,vmax=1)

    axes[0].scatter(lon[~cut],lat[~cut],c='0.7',**kwargs)
    axes[0].scatter(lon[cut],lat[cut],c=prob[cut],**kwargs)
    if kernel is not None:
        plt.sca(axes[0])
        drawKernel(kernel,contour=True,linewidths=2,zorder=0)
        ra,dec = gal2cel(kernel.lon,kernel.lat)
        axes[0].set_xlim(ra-0.4,ra+0.4)
        axes[0].set_ylim(dec-0.4,dec+0.4)

    #axes[0].set_ylabel(r'$\Delta$ DEC (deg)')
    #axes[0].set_xlabel(r'$\Delta$ RA (deg)')
    axes[0].set_xlabel('RA (deg)')
    axes[0].set_ylabel('DEC (deg)')
    axes[0].xaxis.set_major_locator(MaxNLocator(4))
    axes[0].yaxis.set_major_locator(MaxNLocator(4))

    axes[1].errorbar(color[cut],mag[cut],yerr=mag_err_1[cut],fmt='.',c='k',zorder=0.5)
    axes[1].scatter(color[~cut],mag[~cut],c='0.7',**kwargs)
    sc = axes[1].scatter(color[cut],mag[cut],c=prob[cut],**kwargs)

    if isochrone is not None:
        plt.sca(axes[1])
        drawIsochrone(isochrone,lw=25,c='0.5')
        drawIsochrone(isochrone,ls='',marker='.',ms=1,c='k')

    axes[1].set_ylabel(r'$g$')
    axes[1].set_xlabel(r'$g-r$')
    axes[1].set_ylim(config['mag']['max'],config['mag']['min'])
    axes[1].set_xlim(config['color']['min'],config['color']['max'])
    axes[1].xaxis.set_major_locator(MaxNLocator(4))

    divider = make_axes_locatable(axes[1])
    #ax_cb = divider.new_vertical(size="5%", pad=0.05)
    ax_cb = divider.new_horizontal(size="7%", pad=0.1)
    fig.add_axes(ax_cb)
    plt.colorbar(sc, cax=ax_cb, orientation='vertical')
    ax_cb.yaxis.tick_right()
    return fig,axes

def drawIsochrone(isochrone, **kwargs):
    ax = plt.gca()

    kwargs.setdefault('c','0.5')
    kwargs.setdefault('lw',25)
    kwargs.setdefault('zorder',0)
    kwargs.setdefault('ls','-')

    isos = isochrone.isochrones if hasattr(isochrone,'isochrones') else [isochrone]
    for iso in isos:
        logger.debug(iso.filename)
        mass_init, mass_pdf, mass_act, mag_1, mag_2 = iso.sample(mass_steps=5e4)
        mag = mag_1 + isochrone.distance_modulus
        color = mag_1 - mag_2

        # Find discontinuities in the color magnitude distributions
        dmag = np.fabs(mag[1:]-mag[:-1])
        dcolor = np.fabs(color[1:]-color[:-1])
        idx = np.where( (dmag>1.0) | (dcolor>0.25))[0]
        # +1 to map from difference array to original array
        mags = np.split(mag,idx+1)
        colors = np.split(color,idx+1)

        for i,(c,m) in enumerate(zip(colors,mags)):
            msg = '%-4i (%g,%g) -- (%g,%g)'%(i,m[0],c[0],m[-1],c[-1])
            logger.debug(msg)
            if i > 0:
                kwargs['label'] = None
            ax.plot(c,m,**kwargs)

    #ax.invert_yaxis()
    #ax.set_xlim(-0.5,1.5)
    #ax.set_ylim(23,18)

def drawKernel(kernel, contour=False, coords='C', **kwargs):
    ax = plt.gca()

    kwargs.setdefault('cmap',matplotlib.cm.jet)
    kwargs.setdefault('origin','lower')

    ext = kernel.extension
    theta = kernel.theta

    xmin,xmax = -kernel.edge,kernel.edge
    ymin,ymax = -kernel.edge,kernel.edge

    if coords[-1] == 'G':
        lon, lat = kernel.lon, kernel.lat
    elif coords[-1] == 'C':
        lon,lat = gal2cel(kernel.lon, kernel.lat)
    else:
        msg = 'Unrecognized coordinate: %s'%coords
        raise Exception(msg)

    x = np.linspace(xmin,xmax,100)+lon
    y = np.linspace(ymin,ymax,100)+lat
    xx,yy = np.meshgrid(x,y)
    extent = [x[0],x[-1],y[0],y[-1]]
    kwargs.setdefault('extent',extent)
    if coords[-1] == 'C':
        xx,yy = cel2gal(xx,yy)
    
    zz = kernel.pdf(xx.flat,yy.flat).reshape(xx.shape)
    zmax = zz.max()

    if contour:
        levels = 10
        #levels = np.logspace(np.log10(zmax)-1,np.log10(zmax),7)
        ret = ax.contour(zz,levels,**kwargs)
    else:
        val = np.ma.array(zz,mask=zz<zz.max()/100.)
        ret = ax.imshow(val,**kwargs)

    return ret

###################################################

def drawChernoff(ax,ts,bands='smooth',pdf=False,color='r'):
    from scipy.stats import chi2

    logger.debug("Drawing %i simulations..."%len(ts))
    x = plt.linspace(0.1,50,5000)
    bins = np.linspace(-1e-2,50,501)
    centers = (bins[1:]+bins[:-1])/2.

    ax.set_xscale('linear')
    ax.set_yscale('log',nonposy='clip')

    dof = 1
    patches,labels = [],[]
    label = r"$\chi^2_{1} / 2$"
    kwargs = dict(label=label, lw=2, c='k',dashes=(5,2))

    clip_ts = np.where(ts<1e-4, 0, ts)
    if not pdf:
        ax.plot(x,(1-chi2.cdf(x,dof))/2.,**kwargs)
        #fudge = 1/1.4
        #ax.plot(x,(1-chi2.cdf(x,dof))/2.*fudge,**kwargs)
        # Histogram is normalized so first bin = 1 
        n,b,p = ax.hist(clip_ts,cumulative=-1,bins=bins,normed=True,log=True,histtype='step',color=color)
    else:
        num,b = np.histogram(clip_ts,bins=bins)
        c = (b[1:]+b[:-1])/2.
        norm = float(num.sum()*(b[1]-b[0]))
        n = num/norm
        ax.plot(x,(chi2.pdf(x,dof))/2.,**kwargs)
        err = np.sqrt(num)/norm
        yerr = [np.where(err>=n,0.9999*n,err),err]
        # Histogram is normalized so n = num/(len(x)*dbin)
        ax.errorbar(c,n,yerr=yerr,fmt='_',color=color,zorder=0)
        n,b,p = ax.hist(clip_ts,bins=bins,normed=True,log=True,color=color)

    idx = np.argmax(n==0)
    n = n[1:idx]; b=b[1:idx+1]

    ax.set_xlim([0,np.ceil(ts.max())])
    ax.set_ylim([10**np.floor(np.log10(n.min())),1])

    if bands != 'none':
        if bands == 'smooth':
            xvals = np.hstack([b[0],((b[1:]+b[:-1])/2.),b[-1]])
            yvals = np.hstack([n[0],n,n[-1]])
        elif bands == 'sharp':
            xvals = np.repeat(b,2)[1:-1]
            yvals = np.repeat(n,2)
        else:
            msg = 'Unrecognized band type: %s'%bands
            raise Exception(msg)
         
        # Bands...
        err = np.sqrt(yvals/float(len(ts)))
        y_hi = np.clip(yvals+err,1e-32,np.inf)
        y_lo = np.clip(yvals-err,1e-32,np.inf)
         
        #cut = (y_lo > 0)
        kwargs = dict(color='r', alpha='0.5', zorder=0.5)
         
        #ax.fill_between(c[cut], y_lo[cut], y_hi[cut], **kwargs)
        ax.fill_between(xvals, y_lo, y_hi, **kwargs)
        ax.add_patch(plt.Rectangle((0,0),0,0, **kwargs)) # Legend
        
    #ax.annotate(r"$N=%i$"%len(ts), xy=(0.15,0.85), xycoords='axes fraction', 
    #            bbox={'boxstyle':"round",'fc':'1'})

    ax.set_xlabel('TS')
    ax.set_ylabel('PDF' if pdf else 'CDF')

def plotChernoff(ts,bands='smooth',pdf=False):
    fig,ax = plt.subplots(1,1)

    drawChernoff(ax,ts,bands,pdf)


def plot_chain(chain,burn=None,clip=None):
    import triangle
    from ugali.analysis.mcmc import Samples 
    samples = Samples(chain)
    names = samples.names
    results = samples.results(clip=clip,burn=burn)
    truths = [results[n][0] for n in names]
    data = samples[burn:].view((float,len(names)))
    fig = triangle.corner(data, labels=names, truths=truths)
    return fig

###################################################

    
def plotSkymapCatalog(lon,lat,**kwargs):
    """
    Plot a catalog of coordinates on a full-sky map.
    """
    fig = plt.figure()
    ax = plt.subplot(111,projection=projection)
    drawSkymapCatalog(ax,lon,lat,**kwargs)

def drawSkymapCatalog(ax,lon,lat,**kwargs):
    mapping = {
        'ait':'aitoff',
        'mol':'mollweide',
        'lam':'lambert',
        'ham':'hammer'
    }
    kwargs.setdefault('proj','aitoff')
    kwargs.setdefault('s',2)
    kwargs.setdefault('marker','.')
    kwargs.setdefault('c','k')

    proj = kwargs.pop('proj')
    projection = mapping.get(proj,proj)
    #ax.grid()
    # Convert from 
    # [0. < lon < 360] -> [-pi < lon < pi]
    # [-90 < lat < 90] -> [-pi/2 < lat < pi/2]
    lon,lat= np.radians([lon-360.*(lon>180),lat])
    ax.scatter(lon,lat,**kwargs)

def plotSkymap(skymap, proj='mol', **kwargs):
    kwargs.setdefault('xsize',1000)
    if proj.upper() == 'MOL':
        im = healpy.mollview(skymap,**kwargs)
    elif proj.upper() == 'CAR':
        im = healpy.cartview(skymap,**kwargs)
    return im

