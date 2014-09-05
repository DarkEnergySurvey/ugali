"""
Basic plotting tools.
"""
import os
import collections

import matplotlib
try:             os.environ['DISPLAY']
except KeyError: matplotlib.use('Agg')

import numpy
import pylab
import healpy
import pyfits
from mpl_toolkits.axes_grid1 import AxesGrid    

import ugali.utils.config
import ugali.observation.roi
import ugali.observation.catalog
import ugali.utils.skymap
import ugali.utils.projector

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

def projScatter(ax, lon, lat, **kwargs):
    """
    Create a scatter plot on HEALPix projected axes.
    Inputs: lon (deg), lat (deg)
    """
    healpy.projscatter(x,y, lonlat=True, **kwargs)

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
        
    def drawImage(self,ax):
        # Optical Image
        im = ugali.utils.plotting.getSDSSImage(**self.image_kwargs)
        # Flipping JPEG:
        # https://github.com/matplotlib/matplotlib/issues/101
        im =ax.imshow(im[::-1])
        try:
            ax.cax.axis["right"].toggle(ticks=False, ticklabels=False)
            ax.cax.axis["top"].toggle(ticks=False, ticklabels=False)
        except: pass
        ax.annotate("Image",**self.label_kwargs)
        return ax

    def drawStellarDensity(self,ax):
        # Stellar Catalog
        catalog=ugali.observation.catalog.Catalog(self.config,roi=self.roi)
        pix = ugali.utils.projector.angToPix(self.nside, catalog.lon, catalog.lat)
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

    def drawMask(self,ax):
        # MAGLIM Mask
        filenames = self.config.getFilenames()
        catalog_pixels = self.roi.getCatalogPixels()
        mask_map = ugali.utils.skymap.readSparseHealpixMaps(filenames['mask_1'][catalog_pixels], field='MAGLIM')
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

    def drawCMD(self, ax, radius=None, zidx=None):
        import ugali.analysis.isochrone

        if zidx is not None:
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
                ax.scatter(isochrone.color,mag, color='0.5', s=750, zorder=0)
        
        # Stellar Catalog
        catalog = ugali.observation.catalog.Catalog(self.config,roi=self.roi)
        sep = ugali.utils.projector.angsep(self.glon, self.glat, catalog.lon, catalog.lat)
        radius = self.radius if radius is None else radius
        cut = (sep < radius)
        catalog_cmd = catalog.applyCut(cut)
        ax.scatter(catalog_cmd.color, catalog_cmd.mag,color='b')
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


        pix = ugali.utils.projector.angToPix(self.nside, self.glon, self.glat)
        likelihood_pix = ugali.utils.skymap.superpixel(pix,self.nside,self.config.params['coords']['nside_likelihood'])
        config = self.config
        scan = ugali.analysis.scan.Scan(self.config,likelihood_pix)
        likelihood = scan.likelihood
        distance_modulus_array = [self.config.params['likelihood']['distance_modulus_array'][zidx]]
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
        #dirname = self.config.params['output2']['searchdir']
        #basename = self.config.params['output2']['mergefile']
        #filename = os.path.join(dirname,basename)
        filename = self.config.mergefile
        logger.debug("Opening %s..."%filename)
        f = pyfits.open(filename)
        pixels,values = f[1].data['pix'],2*f[1].data['log_likelihood']
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
        axes  = AxesGrid(fig, 111, nrows_ncols = (nrows, ncols),axes_pad=0,label_mode='1', cbar_mode='single',cbar_pad=0,cbar_size='5%',share_all=True,add_all=False)

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
            
            ax.annotate(r"$\mu = %g$"%distances[i],**self.label_kwargs)
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
        glon,glat = self.obj['GLON_MAX'],self.obj['GLAT_MAX']
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
