"""
Basic plotting tools.
"""
import os
import collections
import copy

import matplotlib
try:             os.environ['DISPLAY']
except KeyError: matplotlib.use('Agg')

import yaml
import numpy
import numpy as np
import pylab
import pylab as plt
import healpy
import pyfits
import scipy.ndimage as nd
import scipy.misc

from mpl_toolkits.axes_grid1 import AxesGrid,Grid,ImageGrid, make_axes_locatable
from matplotlib.ticker import MaxNLocator
import mpl_toolkits.axes_grid1.axes_divider as axes_divider

import ugali.utils.config
import ugali.observation.roi
import ugali.observation.catalog
import ugali.utils.skymap
import ugali.utils.projector
import ugali.utils.healpix
import ugali.isochrone

from ugali.utils.healpix import ang2pix
from ugali.utils.projector import mod2dist,gal2cel,cel2gal
from ugali.utils.projector import sphere2image,image2sphere

from ugali.utils.logger import logger

params = {
    #'backend': 'eps',
    'axes.labelsize': 12,
    #'text.fontsize': 12,           
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.major.size': 3,      # major tick size in points
    'xtick.minor.size': 1.5,    # minor tick size in points
    'xtick.major.size': 3,      # major tick size in points
    'xtick.minor.size': 1.5,    # minor tick size in points
    #'text.usetex': True,
    ##'figure.figsize': fig_size,
    #'font.family':'serif',
    #'font.serif':'Computer Modern Roman',
    #'font.size': 10
    }
matplotlib.rcParams.update(params)

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

def drawHealpixMap(map, lon, lat, size=1.0, xsize=501, coord='GC', **kwargs):
    """
    Draw local projection of healpix map.
    """
    ax = plt.gca()
    x = np.linspace(-size,size,xsize)
    y = np.linspace(-size,size,xsize)
    xx, yy = np.meshgrid(x,y)
    
    coord = coord.upper()
        
    if coord == 'GC':
        #Assumes map and (lon,lat) are Galactic, but plotting celestial
        llon, llat = image2sphere(*gal2cel(lon,lat),x=xx.flat,y=yy.flat)
        pix = ang2pix(healpy.get_nside(map),*cel2gal(llon,llat))
    elif coord == 'CG':
        #Assumes map and (lon,lat) are celestial, but plotting Galactic
        llon, llat = image2sphere(*cel2gal(lon,lat),x=xx.flat,y=yy.flat)
        pix = ang2pix(healpy.get_nside(map),*gal2cel(llon,llat))
    else:
        #Assumes plotting the native coordinates
        llon, llat = image2sphere(lon,lat,xx.flat,yy.flat)
        pix = ang2pix(healpy.get_nside(map),llon,llat)

    values = map[pix].reshape(xx.shape)
    zz = np.ma.array(values,mask=(values==healpy.UNSEEN),fill_value=np.nan)

    return drawProjImage(xx,yy,zz,coord=coord,**kwargs)

def drawProjImage(xx, yy, zz=None, coord='C',**kwargs): 
    ax = plt.gca()
    coord = coord.upper()
    if coord[-1] == 'G':
        ax.set_xlabel(r'$\Delta \ell\,(\deg)}$')
        ax.set_ylabel(r'$\Delta b\,(\deg)$')
    elif coord[-1] == 'C':
        ax.set_xlabel(r'$\Delta \alpha_{2000}\,(\deg)$')
        ax.set_ylabel(r'$\Delta \delta_{2000}\,(\deg)$')
    else:
        msg = "Unrecognized coordinate: %"%coord
        logger.warning(msg)
    # Celestial orientation (increasing to the east)
    #ax.set_xlim(xx.max(),xx.min())
    ax.set_xlim(xx.min(),xx.max())
    ax.set_ylim(yy.min(),yy.max())

    if zz is None: return None
    return ax.pcolormesh(xx,yy,zz,**kwargs)
    

############################################################

def getSDSSImage(ra,dec,radius=1.0,xsize=800,opt='GML',**kwargs):
    """
    Download Sloan Digital Sky Survey images
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
    cmd='wget --progress=dot:mega -O %s "%s"'%(tmp.name,url+query)
    subprocess.call(cmd,shell=True)
    im = pylab.imread(tmp.name)
    tmp.close()
    return im


def getDSSImage(ra,dec,radius=1.0,xsize=800,**kwargs):
    """
    Download Digitized Sky Survey images

    https://archive.stsci.edu/cgi-bin/dss_form    
    https://archive.stsci.edu/cgi-bin/dss_search

    Image is in celestial orientation (RA increases to the right)
    https://archive.stsci.edu/dss/script_usage.html

    ra (r) - right ascension
    dec (d) - declination
    equinox (e) - equinox (B1950 or J2000; default: J2000)
    height (h) - height of image (arcminutes; default: 15.0)
    width (w) - width of image (arcminutes; default: 15.0)
    format (f) - image format (FITS or GIF; default: FITS)
    compression (c) - compression (UNIX, GZIP, or NONE; default: NONE; compression 
         applies to FITS only)
    version (v)  - Which version of the survey to use:
         1  - First Generation survey (garden variety)
         2  - Second generation survey (incomplete)
         3  - Check the 2nd generation; if no image is available,
              then go to the 1st generation.
         4  - The Quick V survey (whence came the Guide Stars Catalog;
              used mostly for Phase II proposal submission)
    save (s)  - Save the file to disk instead of trying to display.
         (ON (or anything) or not defined; default: not defined.)
    """
    import subprocess
    import tempfile

    url="https://archive.stsci.edu/cgi-bin/dss_search?"
    scale = 2.0 * radius * 60.
    params=dict(ra='%.3f'%ra,dec='%.3f'%dec,width=scale,height=scale,
                format='gif',version=1)
    #v='poss2ukstu_red'
    query='&'.join("%s=%s"%(k,v) for k,v in params.items())
    
    tmp = tempfile.NamedTemporaryFile(suffix='.gif')
    cmd='wget --progress=dot:mega -O %s "%s"'%(tmp.name,url+query)
    subprocess.call(cmd,shell=True)
    im = pylab.imread(tmp.name)
    tmp.close()
    if xsize: im = scipy.misc.imresize(im,size=(xsize,xsize))
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
        self.coord = 'GC'
        xsize=800
        reso = 60. * 2. * radius / xsize
        self.image_kwargs = dict(ra=self.ra,dec=self.dec,radius=self.radius)
        self.gnom_kwargs = dict(rot=[self.ra,self.dec],reso=reso,xsize=xsize,coord=self.coord,
                                return_projected_map=True,hold=True)
        self.label_kwargs = dict(xy=(0.05,0.05),xycoords='axes fraction', xytext=(0, 0), 
                                 textcoords='offset points',ha='left', va='bottom',size=10,
                                 bbox={'boxstyle':"round",'fc':'1'}, zorder=10)
        
    def _create_catalog(self):
        if hasattr(self,'catalog'): return
        self.catalog = self.get_stars()

    def get_objects(self,select=None):
        config = copy.deepcopy(self.config)
        config['catalog']['selection'] = select
        catalog = ugali.observation.catalog.Catalog(config,roi=self.roi)
        sep = ugali.utils.projector.angsep(self.glon, self.glat, catalog.lon, catalog.lat)
        radius = self.radius*np.sqrt(2)
        cut = (sep < radius)
        return catalog.applyCut(cut)

    def get_stars(self,select=None):
        if hasattr(self,'stars'): return self.stars
        if select is None: select = self.config['catalog']['selection']
        self.stars = self.get_objects(select)
        return self.stars
        
    def get_galaxies(self,select=None):
        if hasattr(self,'galaxies'): return self.galaxies
        if select is not None:
            self.galaxies = self.get_objects(select)
        else:
            catalog = self.get_objects()
            stars = self.get_stars()
            cut = ~np.in1d(catalog.objid,stars.objid)
            self.galaxies = catalog.applyCut(cut)
        return self.galaxies

    def drawSmoothCatalog(self, catalog, label=None, **kwargs):
        ax = plt.gca()
        ra,dec = catalog.ra_dec
        x, y = sphere2image(self.ra,self.dec,ra,dec)

        delta_x = self.radius/100.
        smoothing = 2*delta_x
        bins = numpy.arange(-self.radius, self.radius + 1.e-10, delta_x)
        h, xbins, ybins = numpy.histogram2d(x, y, bins=[bins, bins])
        blur = nd.filters.gaussian_filter(h.T, smoothing / delta_x)

        defaults = dict(cmap='gray_r',rasterized=True)
        kwargs = dict(defaults.items()+kwargs.items())

        xx,yy = np.meshgrid(xbins,ybins)
        im = drawProjImage(xx,yy,blur,coord='C',**kwargs)
        
        if label:
            plt.text(0.05, 0.95, label, fontsize=10, ha='left', va='top', 
                     color='k', transform=pylab.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=1., edgecolor='none'))

    def drawROI(self, ax=None, value=None, pixel=None):
        if not ax: ax = plt.gca()
        roi_map = np.array(healpy.UNSEEN*np.ones(healpy.nside2npix(self.nside)))
        
        if value is None:
            roi_map[self.roi.pixels] = 1
            roi_map[self.roi.pixels_annulus] = 0
            roi_map[self.roi.pixels_target] = 2
        elif value is not None and pixel is None:
            roi_map[self.pixels] = value
        elif value is not None and pixel is not None:
            roi_map[pixel] = value
        else:
            logger.warning('Unable to parse input')
        #im = healpy.gnomview(roi_map,**self.gnom_kwargs)
        im = drawHealpixMap(roi_map,self.glon,self.glat,self.radius,coord=self.coord)
        return im

    def drawImage(self,ax=None,invert=True):
        if not ax: ax = plt.gca()

        if self.config['data']['survey']=='sdss':
            # Optical Image
            im = ugali.utils.plotting.getSDSSImage(**self.image_kwargs)
            # Flipping JPEG:
            # https://github.com/matplotlib/matplotlib/issues/101
            im = im[::-1]
            ax.annotate("SDSS Image",**self.label_kwargs)
        else: 
            im = ugali.utils.plotting.getDSSImage(**self.image_kwargs)
            im = im[::-1,::-1]
            ax.annotate("DSS Image",**self.label_kwargs)

        size=self.image_kwargs.get('radius',1.0)

        # Celestial coordinates
        x = np.linspace(-size,size,im.shape[0])
        y = np.linspace(-size,size,im.shape[1])
        xx, yy = np.meshgrid(x,y)

        #kwargs = dict(cmap='gray',interpolation='none')
        kwargs = dict(cmap='gray',coord='C')
        im = drawProjImage(xx,yy,im,**kwargs)
        
        try: plt.gcf().delaxes(ax.cax)
        except AttributeError: pass
            
        return im

    def drawStellarDensity(self,ax=None):
        if not ax: ax = plt.gca()
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
    
        #im = healpy.gnomview(star_map,**self.gnom_kwargs)
        #healpy.graticule(dpar=1,dmer=1,color='0.5',verbose=False)
        #pylab.close()

        im = drawHealpixMap(star_map,self.glon,self.glat,self.radius,coord=self.coord)
        #im = ax.imshow(im,origin='bottom')
        try:    ax.cax.colorbar(im)
        except: pylab.colorbar(im,ax=ax)
        ax.annotate("Stars",**self.label_kwargs)
        return im

    def drawMask(self,ax=None, mask=None):
        if not ax: ax = plt.gca()
        # MAGLIM Mask
        if mask is None:
            filenames = self.config.getFilenames()
            catalog_pixels = self.roi.getCatalogPixels()
            mask_map = ugali.utils.skymap.readSparseHealpixMaps(filenames['mask_1'][catalog_pixels], field='MAGLIM')
        else:
            mask_map = healpy.UNSEEN*np.ones(healpy.nside2npix(self.config['coords']['nside_pixel']))
            mask_map[mask.roi.pixels] = mask.mask_1.mask_roi_sparse
        mask_map = numpy.where(mask_map == healpy.UNSEEN, 0, mask_map)
         
        #im = healpy.gnomview(mask_map,**self.gnom_kwargs)
        #healpy.graticule(dpar=1,dmer=1,color='0.5',verbose=False)
        #pylab.close()
        #im = ax.imshow(im,origin='bottom')

        im = drawHealpixMap(mask_map,self.glon,self.glat,self.radius,coord=self.coord)

        try: ax.cax.colorbar(im)
        except: pylab.colorbar(im)
        ax.annotate("Mask",**self.label_kwargs)
        return im

    def drawTS(self,ax=None, filename=None, zidx=0):
        if not ax: ax = plt.gca()
        if not filename:
            #dirname = self.config.params['output2']['searchdir']
            #basename = self.config.params['output2']['mergefile']
            #filename = os.path.join(dirname,basename)
            filename = self.config.mergefile

        results=pyfits.open(filename)[1]
        pixels = results.data['PIXEL']
        values = 2*results.data['LOG_LIKELIHOOD']
        if values.ndim == 1: values = values.reshape(-1,1)
        ts_map = healpy.UNSEEN * numpy.ones(healpy.nside2npix(self.nside))
        # Sum through all distance_moduli
        #ts_map[pixels] = values.sum(axis=1)
        # Just at maximum slice from object

        ts_map[pixels] = values[:,zidx]

        #im = healpy.gnomview(ts_map,**self.gnom_kwargs)
        #healpy.graticule(dpar=1,dmer=1,color='0.5',verbose=False)
        #pylab.close()
        #im = ax.imshow(im,origin='bottom')

        im = drawHealpixMap(ts_map,self.glon,self.glat,self.radius,coord=self.coord)

        try: ax.cax.colorbar(im)
        except: pylab.colorbar(im)
        ax.annotate("TS",**self.label_kwargs)
        return im

    def drawCatalog(self, ax=None):
        if not ax: ax = plt.gca()
        # Stellar Catalog
        self._create_catalog()
        healpy.projscatter(self.catalog.lon,self.catalog.lat,c='k',marker='.',lonlat=True,coord=self.gnom_kwargs['coord'])
        ax.annotate("Stars",**self.label_kwargs)

    def drawSpatial(self, ax=None):
        if not ax: ax = plt.gca()
        # Stellar Catalog
        self._create_catalog()
        cut = (self.catalog.color > 0) & (self.catalog.color < 1)
        catalog = self.catalog.applyCut(cut)
        ax.scatter(catalog.lon,catalog.lat,c='k',marker='.',s=1)
        ax.set_xlim(self.glon-0.5,self.glon+0.5)
        ax.set_ylim(self.glat-0.5,self.glat+0.5)
        ax.set_xlabel('GLON (deg)')
        ax.set_ylabel('GLAT (deg)')

    def drawCMD(self, ax=None, radius=None, zidx=None):
        if not ax: ax = plt.gca()
        import ugali.isochrone

        if zidx is not None:
            filename = self.config.mergefile
            logger.debug("Opening %s..."%filename)
            f = pyfits.open(filename)
            distance_modulus = f[2].data['DISTANCE_MODULUS'][zidx]

            iso = ugali.isochrone.Padova(age=12,z=0.0002,mod=distance_modulus)
            #drawIsochrone(iso,ls='',marker='.',ms=1,c='k')
            drawIsochrone(iso)

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


    def drawMembership(self, ax=None, radius=None, zidx=0, mc_source_id=1):
        if not ax: ax = plt.gca()
        import ugali.analysis.scan

        filename = self.config.mergefile
        logger.debug("Opening %s..."%filename)
        f = pyfits.open(filename)
        distance_modulus = f[2].data['DISTANCE_MODULUS'][zidx]

        for ii, name in enumerate(self.config.params['isochrone']['infiles']):
            logger.info('%s %s'%(ii, name))
            isochrone = ugali.isochrone.Isochrone(self.config, name)
            mag = isochrone.mag + distance_modulus
            ax.scatter(isochrone.color,mag, color='0.5', s=800, zorder=0)


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
        pixels,values = f[1].data['PIXEL'],2*f[1].data['LOG_LIKELIHOOD']
        if values.ndim == 1: values = values.reshape(-1,1)
        distances = f[2].data['DISTANCE_MODULUS']
        if distances.ndim == 1: distances = distances.reshape(-1,1)
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
        fig = pylab.figure(figsize=(8,8))
        axes = AxesGrid(fig, 111,nrows_ncols = (2, 2),axes_pad=0.25,
                        cbar_mode='each',cbar_pad=0,cbar_size='5%',
                        share_all=True,aspect=True,
                        label_mode='L')

        #fig,axes = plt.subplots(2,2)
        #axes = axes.flatten()

        #for ax in axes:
        #    ax.get_xaxis().set_visible(False)
        #    ax.get_yaxis().set_visible(False)

        #plt.sca(axes[0]); self.drawImage(axes[0])
        #plt.sca(axes[1]); self.drawStellarDensity(axes[1])
        #plt.sca(axes[2]); self.drawMask(axes[2])
        #plt.sca(axes[3]); self.drawTS(axes[3])
        try: plt.sca(axes[0]); self.drawImage()
        except IOError as e: logger.warn(str(e))
            
        plt.sca(axes[1]); self.drawStellarDensity()
        plt.sca(axes[2]); self.drawMask()
        try: plt.sca(axes[3]); self.drawTS()
        except IOError as e: logger.warn(str(e))
            
        axes[0].set_xlim(self.radius,-self.radius)
        axes[0].set_ylim(-self.radius,self.radius)

        return fig,axes

    plot = plot3


class ObjectPlotter(BasePlotter):
    """ For plotting 'Objects' identified through candidate search. """

    def __init__(self,obj,config,radius=1.0):
        self.obj = obj
        glon,glat = self.obj['GLON'],self.obj['GLAT']
        super(ObjectPlotter,self).__init__(glon,glat,config,radius)
        self.set_zidx()

    def set_zidx(self):
        names = [n.upper() for n in self.obj.array.dtype.names]
        mod = np.array(self.config['scan']['distance_modulus_array'])
        if 'ZIDX_MAX' in names:
            self.zidx = self.obj['ZIDX_MAX'] 
        elif 'DISTANCE_MODULUS' in names:
            dist_mod = self.obj['DISTANCE_MODULUS']
            self.zidx = np.abs(mod - dist_mod).argmin()
        elif 'MODULUS' in names:
            dist_mod = self.obj['MODULUS']
            self.zidx = np.abs(mod - dist_mod).argmin()
        elif 'DISTANCE' in names:
            dist_mod = mod2dist(self.obj['DISTANCE'])
            self.zidx = np.argmax((mod - dist_mod) > 0)
        else:
            msg = "Failed to parse distance index"
            raise Exception(msg)

    def drawTS(self, filename=None, zidx=None):
        ax = plt.gca()
        if zidx is None: zidx = self.zidx
        super(ObjectPlotter,self).drawTS(ax,filename,zidx)

    def drawCMD(self, radius=None, zidx=None):
        ax = plt.gca()
        if zidx is None: zidx = self.zidx
        super(ObjectPlotter,self).drawCMD(ax,radius,zidx)

    def drawMembership(self, radius=None, zidx=None, mc_source_id=1):
        ax = plt.gca()
        if zidx is None: zidx = self.zidx
        super(ObjectPlotter,self).drawMembership(ax,radius,zidx,mc_source_id)


class SourcePlotter(BasePlotter):
    """ For plotting 'Objects' identified through candidate search. """

    def __init__(self,source,config,radius=1.0):
        glon,glat = source.lon,source.lat
        super(SourcePlotter,self).__init__(glon,glat,config,radius)
        #self.select = self.config['catalog'].pop('selection')

        self.source = source
        self.isochrone = self.source.isochrone
        self.kernel = self.source.kernel
        self.set_zidx()

    def isochrone_selection(self,catalog,dist=0.1):
        # Cookie cutter
        return cutIsochronePath(catalog.mag_1, catalog.mag_2, 
                                catalog.mag_err_1, catalog.mag_err_2, 
                                self.isochrone, radius=dist)
        
    def set_zidx(self):
        mod = np.array(self.config['scan']['distance_modulus_array'])
        dist_mod = self.isochrone.distance_modulus
        self.zidx = np.abs(mod - dist_mod).argmin()

    def drawSmoothStars(self,**kwargs):
        stars = self.get_stars()
        sel = self.isochrone_selection(stars,dist=0.1)
        self.drawSmoothCatalog(stars.applyCut(sel),'Filtered Stars',**kwargs)

    def drawSmoothGalaxies(self,**kwargs):
        galaxies = self.get_galaxies()
        sel = self.isochrone_selection(galaxies,dist=0.1)
        self.drawSmoothCatalog(galaxies.applyCut(sel),'Filtered Galaxies',**kwargs)

    def drawHessDiagram(self,catalog=None):
        ax = plt.gca()
        if not catalog: catalog = self.get_stars()

        r_peak = self.kernel.extension
        angsep = ugali.utils.projector.angsep(self.ra, self.dec, catalog.ra, catalog.dec)
        cut_inner = (angsep < r_peak)
        cut_annulus = (angsep > 0.5) & (angsep < 1.) # deg

        mmin, mmax = 16., 24.
        cmin, cmax = -0.5, 1.0
        mbins = np.linspace(mmin, mmax, 150)
        cbins = np.linspace(cmin, cmax, 150)

        color = catalog.color[cut_annulus]
        mag = catalog.mag[cut_annulus]

        h, xbins, ybins = numpy.histogram2d(color, mag, bins=[cbins,mbins])
        blur = nd.filters.gaussian_filter(h.T, 2)
        kwargs = dict(extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
                      cmap='gray_r', aspect='auto', origin='lower', 
                      rasterized=True, interpolation='none')
        ax.imshow(blur, **kwargs)

        pylab.scatter(catalog.color[cut_inner], catalog.mag[cut_inner], 
                      c='red', s=7, edgecolor='none')# label=r'$r < %.2f$ deg'%(r_peak))
        ugali.utils.plotting.drawIsochrone(self.isochrone, c='b', zorder=10)
        ax.set_xlim(-0.5, 1.)
        ax.set_ylim(24., 16.)
        plt.xlabel(r'$g - r$')
        plt.ylabel(r'$g$')
        plt.xticks([-0.5, 0., 0.5, 1.])
        plt.yticks(numpy.arange(mmax - 1., mmin - 1., -1.))

        radius_string = (r'${\rm r}<%.1f$ arcmin'%( 60 * r_peak))
        pylab.text(0.05, 0.95, radius_string, 
                   fontsize=10, ha='left', va='top', color='red', 
                   transform=pylab.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=1., edgecolor='none'))


    def drawMembersSpatial(self,data):
        ax = plt.gca()
        if isinstance(data,basestring):
            filename = data
            data = pyfits.open(filename)[1].data

        xmin, xmax = -0.25,0.25
        ymin, ymax = -0.25,0.25
        xx,yy = np.meshgrid(np.linspace(xmin,xmax),np.linspace(ymin,ymax))

        x_prob, y_prob = sphere2image(self.ra, self.dec, data['RA'], data['DEC'])

        sel = (x_prob > xmin)&(x_prob < xmax) & (y_prob > ymin)&(y_prob < ymax)
        sel_prob = data['PROB'][sel] > 5.e-2
        index_sort = numpy.argsort(data['PROB'][sel][sel_prob])

        plt.scatter(x_prob[sel][~sel_prob], y_prob[sel][~sel_prob], 
                      marker='o', s=2, c='0.75', edgecolor='none')
        sc = plt.scatter(x_prob[sel][sel_prob][index_sort], 
                         y_prob[sel][sel_prob][index_sort], 
                         c=data['PROB'][sel][sel_prob][index_sort], 
                         marker='o', s=10, edgecolor='none', cmap='jet', vmin=0., vmax=1.) # Spectral_r

        drawProjImage(xx,yy,None,coord='C')

        #ax.set_xlim(xmax, xmin)
        #ax.set_ylim(ymin, ymax)
        #plt.xlabel(r'$\Delta \alpha_{2000}\,(\deg)$')
        #plt.ylabel(r'$\Delta \delta_{2000}\,(\deg)$')
        plt.xticks([-0.2, 0., 0.2])
        plt.yticks([-0.2, 0., 0.2])

        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="7%", pad=0.1)
        plt.gcf().add_axes(ax_cb)
        pylab.colorbar(sc, cax=ax_cb, orientation='vertical', ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], label='Membership Probability')
        ax_cb.yaxis.tick_right()

    def drawMembersCMD(self,data):
        ax = plt.gca()
        if isinstance(data,basestring):
            filename = data
            data = pyfits.open(filename)[1].data

        xmin, xmax = -0.25,0.25
        ymin, ymax = -0.25,0.25
        mmin, mmax = 16., 24.
        cmin, cmax = -0.5, 1.0
        mbins = np.linspace(mmin, mmax, 150)
        cbins = np.linspace(cmin, cmax, 150)

        mag_1 = data[self.config['catalog']['mag_1_field']]
        mag_2 = data[self.config['catalog']['mag_2_field']]

        x_prob, y_prob = sphere2image(self.ra, self.dec, data['RA'], data['DEC'])

        sel = (x_prob > xmin)&(x_prob < xmax) & (y_prob > ymin)&(y_prob < ymax)
        sel_prob = data['PROB'][sel] > 5.e-2
        index_sort = numpy.argsort(data['PROB'][sel][sel_prob])

        plt.scatter(data['COLOR'][sel][~sel_prob], mag_1[sel][~sel_prob],
              marker='o',s=2,c='0.75',edgecolor='none')
        sc = pylab.scatter(data['COLOR'][sel][sel_prob][index_sort], mag_1[sel][sel_prob][index_sort], 
                   c=data['PROB'][sel][sel_prob][index_sort], 
                   marker='o', s=10, edgecolor='none', cmap='jet', vmin=0., vmax=1) 
        pylab.xlim(cmin, cmax)
        pylab.ylim(mmax, mmin)
        pylab.xlabel(r'$g - r$')
        pylab.ylabel(r'$g$')
        #axes[1].yaxis.set_major_locator(MaxNLocator(prune='lower'))
        pylab.xticks([-0.5, 0., 0.5, 1.])
        pylab.yticks(numpy.arange(mmax - 1., mmin - 1., -1.))

        ugali.utils.plotting.drawIsochrone(self.isochrone, c='k', zorder=10)

        pylab.text(0.05, 0.95, r'$\Sigma p_{i} = %i$'%(data['PROB'].sum()),
                   fontsize=10, horizontalalignment='left', verticalalignment='top', color='k', transform=pylab.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=1., edgecolor='none'))

        divider = make_axes_locatable(pylab.gca())
        ax_cb = divider.new_horizontal(size="7%", pad=0.1)
        plt.gcf().add_axes(ax_cb)
        pylab.colorbar(sc, cax=ax_cb, orientation='vertical', ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], label='Membership Probability')
        ax_cb.yaxis.tick_right()

    def drawDensityProfile(self, catalog=None):

        rmax = 24. # arcmin
        bins = numpy.arange(0, rmax + 1.e-10, 2.)
        centers = 0.5 * (bins[1:] + bins[0:-1])
        area = numpy.pi * (bins[1:]**2 - bins[0:-1]**2)

        r_peak = self.kernel.extension 

        stars = self.get_stars()
        angsep = ugali.utils.projector.angsep(self.ra, self.dec, 
                                              stars.ra, stars.dec)

        angsep_arcmin = angsep * 60 # arcmin
        cut_iso = self.isochrone_selection(stars)
        h = numpy.histogram(angsep_arcmin[(angsep_arcmin < rmax) & cut_iso], bins=bins)[0]
        h_out = numpy.histogram(angsep_arcmin[(angsep_arcmin < rmax) & (~cut_iso)], bins=bins)[0]

        gals = self.get_galaxies()
        if len(gals):
            angsep_gal = ugali.utils.projector.angsep(self.ra, self.dec, 
                                              gals.ra, gals.dec)

            angsep_gal_arcmin = angsep_gal * 60 # arcmin
            cut_iso_gal = self.isochrone_selection(gals)
            h_gal = np.histogram(angsep_gal_arcmin[(angsep_gal_arcmin < rmax) & cut_iso_gal], bins=bins)[0]
            h_gal_out = np.histogram(angsep_gal_arcmin[(angsep_gal_arcmin < rmax) & (~cut_iso_gal)], bins=bins)[0]

        plt.plot(centers, h/area, c='red', label='Filtered Stars')
        plt.errorbar(centers, h/area, yerr=(numpy.sqrt(h) / area), ecolor='red', c='red')
        plt.scatter(centers, h/area, edgecolor='none', c='red', zorder=22)

        plt.plot(centers, h_out/area, c='gray', label='Unfiltered Stars')
        plt.errorbar(centers, h_out/area, yerr=(numpy.sqrt(h_out) / area), ecolor='gray', c='gray')
        plt.scatter(centers, h_out/area, edgecolor='none', c='gray', zorder=21)

        if len(gals):
            plt.plot(centers, h_gal/area, c='black', label='Filtered Galaxies')
            plt.errorbar(centers, h_gal/area, yerr=(numpy.sqrt(h_gal) / area), ecolor='black', c='black')
            plt.scatter(centers, h_gal/area, edgecolor='none', c='black', zorder=20)

        plt.xlabel('Angular Separation (arcmin)')
        plt.ylabel(r'Density (arcmin$^{-2}$)')
        plt.xlim(0., rmax)
        ymax = pylab.ylim()[1]
        #pylab.ylim(0, ymax)
        pylab.ylim(0, 12)
        pylab.legend(loc='upper right', frameon=False, fontsize=10)


    def plot6(self, filename, title=None):
        fig = plt.figure('summary', figsize=(11, 6))
        fig.subplots_adjust(wspace=0.4, hspace=0.25)
        fdg = r'{.}\!^\circ'
        coordstring = ('%.2f, %.2f'%(self.ra, self.dec)).replace('.',fdg)
        if title is None:
            #title = r'%s; ($\alpha_{2000}$, $\delta_{2000}$, $m-M$) = (%s, %.2f)'%(self.source.name, coordstring, self.isochrone.distance_modulus)
            title = r'$(\alpha_{2000}, \delta_{2000}, m-M) = (%s, %.1f)$'%(coordstring, self.isochrone.distance_modulus)

        if title: 
            plt.suptitle(title, fontsize=14)
        
        logger.debug("Drawing smooth stars...")
        plt.subplot(2, 3, 1)
        self.drawSmoothStars()

        logger.debug("Drawing density profile...")
        pylab.subplot(2, 3, 2)
        self.drawDensityProfile()
         
        logger.debug("Drawing spatial distribution of members...")
        pylab.subplot(2, 3, 3)
        self.drawMembersSpatial(filename)

        logger.debug("Drawing smooth galaxies...")
        plt.subplot(2, 3, 4)
        self.drawSmoothGalaxies()

        logger.debug("Drawing Hess diagram...")         
        plt.subplot(2,3,5)
        self.drawHessDiagram()

        logger.debug("Drawing CMD of members...")                  
        pylab.subplot(2, 3, 6)
        self.drawMembersCMD(filename)

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


def draw_slices(hist, func=np.sum, **kwargs):
    """ Draw horizontal and vertical slices through histogram """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    kwargs.setdefault('ls','-')
    ax = plt.gca()

    data = hist

    # Slices
    vslice = func(data,axis=0)
    hslice = func(data,axis=1)

    npix = np.array(data.shape)
    #xlim,ylim = plt.array(zip([0,0],npix-1))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    #extent = ax.get_extent()
    #xlim =extent[:2]
    #ylim = extent[2:]

    # Bin centers
    xbin = np.linspace(xlim[0],xlim[1],len(vslice))#+0.5 
    ybin = np.linspace(ylim[0],ylim[1],len(hslice))#+0.5
    divider = make_axes_locatable(ax)
    
    #gh2 = pywcsgrid2.GridHelperSimple(wcs=self.header, axis_nums=[2, 1])
    hax = divider.append_axes("right", size=1.2, pad=0.05,sharey=ax,
                              axes_class=axes_divider.LocatableAxes)
    hax.axis["left"].toggle(label=False, ticklabels=False)
    #hax.plot(hslice, plt.arange(*ylim)+0.5,'-') # Bin center
    hax.plot(hslice, ybin, **kwargs) # Bin center
    hax.xaxis.set_major_locator(MaxNLocator(4,prune='both'))
    hax.set_ylim(*ylim)

    #gh1 = pywcsgrid2.GridHelperSimple(wcs=self.header, axis_nums=[0, 2])
    vax = divider.append_axes("top", size=1.2, pad=0.05, sharex=ax,
                              axes_class=axes_divider.LocatableAxes)
    vax.axis["bottom"].toggle(label=False, ticklabels=False)
    vax.plot(xbin, vslice, **kwargs) 
    vax.yaxis.set_major_locator(MaxNLocator(4,prune='lower'))
    vax.set_xlim(*xlim)

    return vax,hax

def draw_sum_slices(hist, **kwargs):
    return draw_slices(hist,func=np.sum, **kwargs)

def draw_max_slices(hist, **kwargs):
    return draw_slices(hist,func=np.max, **kwargs)

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

def plotMembership(config, data=None, kernel=None, isochrone=None, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    config = ugali.utils.config.Config(config)
    if isinstance(data,basestring):
        hdu = pyfits.open(data)[1]
        data = hdu.data
        header = hdu.header

    defaults = dict(s=20,edgecolor='none',vmin=0,vmax=1,zorder=3)
    kwargs = dict(defaults.items()+kwargs.items())

    bkg_kwargs = dict(s=3,zorder=0,c='0.70')
    bkg_kwargs = dict(kwargs.items()+bkg_kwargs.items())

    try: 
        sort = np.argsort(data['PROB'])
        prob = data['PROB'][sort]
    except:
        prob = np.zeros(len(data['RA']))+1

    lon,lat = data['RA'][sort],data['DEC'][sort]
    
    lon0,lat0 = np.median(lon),np.median(lat)
    x,y = sphere2image(lon0,lat0,lon,lat)
    lon0,lat0 = image2sphere(lon0,lat0,(x.max()+x.min())/2.,(y.max()+y.min())/2.)
    lon,lat = sphere2image(lon0,lat0,lon,lat)

    color = data['COLOR'][sort]
    cut = (prob > 0.01)

    # ADW: Sometimes may be mag_2
    mag = data[config['catalog']['mag_1_field']][sort]
    mag_err_1 = data[config['catalog']['mag_err_1_field']][sort]
    mag_err_2 = data[config['catalog']['mag_err_2_field']][sort]

    fig,axes = plt.subplots(1,2,figsize=(10,5))
        
    #proj = ugali.utils.projector.Projector(np.median(lon),np.median(lat))
    #x,y = proj.sphereToImage(lon,lat)
    #sc = axes[0].scatter(x,y,c=prob,vmin=0,vmax=1)

    axes[0].scatter(lon[~cut],lat[~cut],**bkg_kwargs)
    axes[0].scatter(lon[cut],lat[cut],c=prob[cut],**kwargs)
    #if kernel is not None:
    #    plt.sca(axes[0])
    #    k = copy.deepcopy(kernel)
    #    levels=[0,k._pdf(k.extension),np.inf]
    #    k.lon,k.lat = cel2gal(0,0)
    #    drawKernel(k,contour=True,linewidths=2,zorder=0,levels=levels)

    #axes[0].set_xlim(lon0-0.4,lon0+0.4)
    #axes[0].set_ylim(lat0-0.4,lat0+0.4)
    #axes[0].set_xlabel('RA (deg)')
    #axes[0].set_ylabel('DEC (deg)')

    axes[0].set_xlim(lon.min(),lon.max())
    axes[0].set_ylim(lat.min(),lat.max())
    axes[0].set_ylabel(r'$\Delta$ DEC (deg)')
    axes[0].set_xlabel(r'$\Delta$ RA (deg)')

    axes[0].xaxis.set_major_locator(MaxNLocator(4))
    axes[0].yaxis.set_major_locator(MaxNLocator(4))
    axes[0].invert_xaxis()

    axes[1].errorbar(color[cut],mag[cut],yerr=mag_err_1[cut],fmt='.',c='k',zorder=0.5)
    axes[1].scatter(color[~cut],mag[~cut],**bkg_kwargs)
    sc = axes[1].scatter(color[cut],mag[cut],c=prob[cut],**kwargs)

    if isochrone is not None:
        plt.sca(axes[1])
        drawIsochrone(isochrone,cookie=False)

    axes[1].set_ylabel(r'$g$')
    axes[1].set_xlabel(r'$g-r$')
    axes[1].set_ylim(config['mag']['max'],config['mag']['min'])
    axes[1].set_xlim(config['color']['min'],config['color']['max'])
    axes[1].xaxis.set_major_locator(MaxNLocator(4))

    try: 
        divider = make_axes_locatable(axes[1])
        #ax_cb = divider.new_vertical(size="5%", pad=0.05)
        ax_cb = divider.new_horizontal(size="7%", pad=0.1)
        fig.add_axes(ax_cb)
        plt.colorbar(sc, cax=ax_cb, orientation='vertical')
        ax_cb.yaxis.tick_right()
    except:
        logger.warning("No colorbar")
    return fig,axes

def drawIsochrone(isochrone, **kwargs):
    ax = plt.gca()
    logger.debug(str(isochrone))
    if kwargs.pop('cookie',None):
        # Broad cookie cutter
        defaults = dict(alpha=0.5, color='0.5', zorder=0, 
                        linewidth=15, linestyle='-')
    else:
        # Thin lines
        defaults = dict(color='k', linestyle='-')
    kwargs = dict(defaults.items()+kwargs.items())

    isos = isochrone.isochrones if hasattr(isochrone,'isochrones') else [isochrone]
    for iso in isos:
        iso = copy.deepcopy(iso)
        logger.debug(iso.filename)
        iso.hb_spread = False
        mass_init,mass_pdf,mass_act,mag_1,mag_2 = iso.sample(mass_steps=1e3)
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
    return ax

def drawKernel(kernel, contour=False, coords='C', **kwargs):
    ax = plt.gca()

    if 'colors' not in kwargs:
        kwargs.setdefault('cmap',matplotlib.cm.jet)
    kwargs.setdefault('origin','lower')

    ext   = kernel.extension
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

    x = np.linspace(xmin,xmax,500)+lon
    y = np.linspace(ymin,ymax,500)+lat
    xx,yy = np.meshgrid(x,y)
    extent = [x[0],x[-1],y[0],y[-1]]
    kwargs.setdefault('extent',extent)

    if coords[-1] == 'C': xx,yy = cel2gal(xx,yy)
    
    zz = kernel.pdf(xx.flat,yy.flat).reshape(xx.shape)
    zmax = zz.max()

    if contour:
        levels = kwargs.pop('levels',10)
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
    #import triangle
    import corner
    from ugali.analysis.mcmc import Samples 
    samples = Samples(chain)
    names = samples.names
    results = samples.results(clip=clip,burn=burn)
    truths = [results[n][0] for n in names]
    data = samples[burn:].view((float,len(names)))
    fig = corner.corner(data, labels=names, truths=truths)
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

def plotTriangle(srcfile,samples,burn=0,**kwargs):
    #import triangle
    import corner
    import ugali.analysis.source
    import ugali.analysis.mcmc
    #matplotlib.rcParams.update({'text.usetex': True})

    source = ugali.analysis.source.Source()
    source.load(srcfile,section='source')
    params = source.get_params()
    results = yaml.load(open(srcfile))['results']
    samples = ugali.analysis.mcmc.Samples(samples)

    names = samples.names
    labels = names 
    truths = [params[n] for n in names]
    chain = samples.get(burn=burn,clip=5)

    ### Triangle plot
    #extents = [[0,15e3],[323.6,323.8],[-59.8,-59.7],[0,0.1],[19.5,20.5]]
    
    kwargs.setdefault('extents',None)
    kwargs.setdefault('plot_contours',True)
    kwargs.setdefault('plot_datapoints',True)
    kwargs.setdefault('verbose',False)
    kwargs.setdefault('quantiles',[0.16,0.84])

    if len(names) > 1:
        fig = corner.corner(chain,labels=labels,truths=truths,**kwargs)
    else:
        fig = plt.figure()
        plt.hist(chain,bins=100)
        plt.xlabel(names[0])
        
    try:
        text  = 'RA,DEC = (%.2f,%.2f)\n'%(results['ra'][0],results['dec'][0])
        text += '(m-M,D) = (%.1f, %.0f kpc)\n'%(results['distance_modulus'][0],results['distance'][0])
        text += r'$r_h$ = %.1f arcmin'%(results['extension_arcmin'][0])+'\n'
        text += 'TS = %.1f\n'%results['ts'][0]
        text += 'NSamples = %i\n'%(len(chain))
        #plt.figtext(0.65,0.90,text,ha='left',va='top')
    except KeyError as e:
        logger.warning(str(e))
        pass

    label = map(str.capitalize,source.name.split('_'))
    label[-1] = label[-1].upper()
    title = '%s'%' '.join(label)
    plt.suptitle(title)


############################################################

def makePath(x_path, y_path, epsilon=1.e-10):
    """
    Create closed path.
    """
    x_path_closed = numpy.concatenate([x_path, x_path[::-1]])
    y_path_closed = numpy.concatenate([y_path, epsilon + y_path[::-1]])
    path = matplotlib.path.Path(zip(x_path_closed, y_path_closed))
    return path

############################################################

def cutIsochronePath(g, r, g_err, r_err, isochrone, radius=0.1, return_all=False):
    """
    Cut to identify objects within isochrone cookie-cutter.

    ADW: This should be moved into the isochrone class.
    """
    import scipy.interpolate
    from ugali.isochrone import CompositeIsochrone

    if isinstance(isochrone, CompositeIsochrone):
        isochrone = isochrone.isochrones[0]

    if len(g) == 0:
        return np.array([],dtype=bool)

    try:
        if numpy.all(isochrone.stage == 'Main'):
            # Dotter case
            index_transition = len(isochrone.stage)
        else:
            # Other cases
            index_transition = numpy.nonzero(isochrone.stage > 3)[0][0] + 1    
    except AttributeError:
        index_transition = 1

    mag_1_rgb = isochrone.mag_1[0: index_transition] + isochrone.distance_modulus
    mag_2_rgb = isochrone.mag_2[0: index_transition] + isochrone.distance_modulus
    mag_1_rgb = mag_1_rgb[::-1]
    mag_2_rgb = mag_2_rgb[::-1]
    
    # Cut one way...
    f_isochrone = scipy.interpolate.interp1d(mag_2_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = numpy.fabs((g - r) - f_isochrone(r))
    cut_2 = (color_diff < numpy.sqrt(0.1**2 + r_err**2 + g_err**2))

     # ...and now the other
    f_isochrone = scipy.interpolate.interp1d(mag_1_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = numpy.fabs((g - r) - f_isochrone(g))
    cut_1 = (color_diff < numpy.sqrt(0.1**2 + r_err**2 + g_err**2))

    cut = numpy.logical_or(cut_1, cut_2)
    
    # If using Padova isochrone, also include horizontal branch
    if not numpy.all(isochrone.stage == 'Main'):
        index_transition = numpy.nonzero(isochrone.stage > 3)[0][0] + 1
        mag_1_hb = isochrone.mag_1[index_transition:] + isochrone.distance_modulus
        mag_2_hb = isochrone.mag_2[index_transition:] + isochrone.distance_modulus
        path_hb = makePath(mag_1_hb, mag_2_hb)
        cut_hb = path_hb.contains_points(zip(g, r), radius=0.1)
        logger.debug('Applying HB selection')
        logger.debug(numpy.sum(cut))
        cut = numpy.logical_or(cut, cut_hb)
        logger.debug(numpy.sum(cut))

    mag_bins = numpy.arange(16., 24.1, 0.1)
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[0:-1])
    magerr = numpy.tile(0., len(mag_centers))
    for ii in range(0, len(mag_bins) - 1):
        cut_mag_bin = (g > mag_bins[ii]) & (g < mag_bins[ii + 1])
        magerr[ii] = numpy.median(numpy.sqrt(0.1**2 + r_err[cut_mag_bin]**2 + g_err[cut_mag_bin]**2))

    if return_all:
        return cut, mag_centers[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) + magerr)[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) - magerr)[f_isochrone(mag_centers) < 100]
    else:
        return cut

############################################################
