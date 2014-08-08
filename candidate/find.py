#!/usr/bin/env python
import healpy
import pyfits
import numpy as np
import pylab as plt
import scipy.ndimage as ndimage
import copy
import subprocess
import os

import ugali.utils.skymap
import ugali.utils.config
import ugali.association.sources 
from ugali.utils.logger import logger
from ugali.utils.binning import reverseHistogram
from ugali.utils.projector import pixToAng,galToCel
from ugali.utils.projector import Projector


class CandidateSearch(object):
    """
    Object used to search for candidate objects in TS maps.
    """
    def __init__(self, config, filename=None):
        self.config = config
        if filename is None:
            dirname = config['output']['savedir_results']
            os.path.join(dirname,"likelihood_merged.fits")
        self.filename = filename
        self._load()
        self._config()

    def _config(self):
        self.nside = self.config.params['coords']['nside_pixel']
        self.threshold = self.config.params['search']['threshold'] # = 10
        self.xsize = self.config.params['search']['xsize'] # = 10000
        self.minpix = self.config.params['search']['minpix'] # = 1
        self.labelfile = os.path.join(self.config['output']['results_savedir'],self.config['output']['labelfile'])
        self.objectfile = os.path.join(self.config['output']['results_savedir'],self.config['output']['objectfile'])
        
    def _load(self):
        f = pyfits.open(self.filename)
        self.pixels = f[1].data['pix']
        self.values = f[1].data['log_likelihood']
        self.distances = f[2].data['DISTANCE_MODULUS']

    def createLabels(self):
        kwargs=dict(pixels=self.pixels,values=self.values,nside=self.nside,
                    threshold=self.threshold,xsize=self.xsize)
        self.labels,self.nlabels = CandidateSearch.labelHealpix(**kwargs)
        return self.labels, self.nlabels

    def writeLabels(self,filename=None):
        if filename is None: filename = self.labelfile
        data_dict = {'LABEL':labels}
        ugali.utils.skymap.writeSparseHealpixMap(self.pixels,data_dict,self.nside,filename,
                                                 distance_modulus_array=self.distance)

    def loadLabels(self,filename=None):
        if filename is None: filename = self.labelfile
        f = pyfits.open(self.filename)
        if not (self.pixels == f[1].data['pix']).all(): 
            raise Exception("...")
        if not (self.distances == f[2].data['DISTANCE_MODULUS']).all():
            raise Exception("...")            

        self.labels = f[1].data['label']
        self.nlabels = numpy.unique(self.labels)
        return self.labels, self.nlabels
        
    def createObjects(self):
        hist,edges,rev = reverseHistogram(self.labels,bins=np.arange(self.nlabels+2))
        self.rev = rev
        # Make some cut on the minimum size of a labelled object
        good, = np.where( (hist >= minpix) )
        # Get rid of zero label (below threshold)
        self.good, = np.nonzero(good)

        kwargs=dict(pixels=self.pixels,values=self.values,nside=self.nside, 
                    zvalues=self.distances, rev=self.rev, good=self.good)

        self.objects = CandidateSearch.findObjects(**kwargs)
        return self.objects

    def writeObjects(self,filename=None):
        pass

    @staticmethod
    def labelHealpix(pixels, values, nside, threshold=0, xsize=1000):
        """
        Label contiguous regions of a (sparse) HEALPix map. Works by mapping 
        HEALPix array to a Mollweide projection and applying scipy.ndimage.label
     
        Assumes non-nested HEALPix map.
        
        Parameters:
        pixels    : Pixel values associated to (sparse) HEALPix array
        values    : (Sparse) HEALPix array of data values
        nside     : HEALPix dimensionality
        threshold : Threshold value for object detection
        xsize     : Size of Mollweide projection
        
        Returns:
        labels, nlabels
        """
        proj = healpy.projector.MollweideProj(xsize=xsize)
        vec = healpy.pix2vec(nside,pixels)
        xy = proj.vec2xy(vec)
        ij = proj.xy2ij(xy)
        xx,yy = proj.ij2xy()
     
        # Convert to Mollweide
        searchims = []
        if values.ndim < 2: iterate = [values]
        else:               iterate = values.T
        for i,value in enumerate(iterate):
            print i
            searchim = np.zeros(xx.shape,dtype=bool)
            select = (value > threshold)
            yidx = ij[0][select]; xidx = ij[1][select]
            searchim[yidx,xidx] |= True
            searchims.append( searchim )
     
        # Do the labeling
        searchims = np.array(searchims)
        s = ndimage.generate_binary_structure(searchims.ndim,searchims.ndim)
        labels,nlabels = ndimage.label(searchims,structure=s)
     
        # Convert back to healpix
        pix_labels = labels[:,ij[0],ij[1]].T
        pix_labels *= (values > threshold) # re-trim
        return pix_labels, nlabels

    @staticmethod
    def findObjects(pixels, values, nside, zvalues, rev, good):
        """
        Characterize labelled candidates in a multi-dimensional HEALPix map.
     
        Parameters:
        values    : (Sparse) HEALPix array of data values
        nside     : HEALPix dimensionality
        pixels    : Pixel values associated to (sparse) HEALPix array
        zvalues   : Values of the z-dimension (usually distance modulus)
        rev       : Reverse indices for pixels in each "island"
        good      : Array containg labels for each "island"
     
        Returns:
        objs      : numpy.recarray of object characteristics
        """
     
        ngood = len(good)
        objs = np.recarray((ngood,),
                           dtype=[('LABEL','i4'),
                                  ('NPIX','i4'),
                                  ('VALUE_MAX','f4'),
                                  ('IDX_MAX','i4'),
                                  ('ZIDX_MAX','i4'),
                                  ('PIX_MAX','i4'),
                                  ('GLON_MAX','f4'),
                                  ('GLAT_MAX','f4'),
                                  ('ZVAL_MAX','f4'),
                                  ('GLON_CENT','f4'),
                                  ('GLAT_CENT','f4'),
                                  ('ZVAL_CENT','f4'),
                                  ('GLON_BARY','f4'),
                                  ('GLAT_BARY','f4'),
                                  ('ZVAL_BARY','f4'),
                                  ('CUT','i2'),])
        objs['CUT'][:] = 0
     
        shape = values.shape
        ncol = shape[1]
        for i in range(0,ngood):
            logger.debug("i=%i",i)
            # This code could use some cleanup...
            indices=rev[rev[good[i]]:rev[good[i]+1]]
            npix = len(indices)
            idx = indices // ncol # This is the spatial index
            zidx = indices % ncol  # This is the distance index
     
            pix = pixels[idx] # This is the healpix pixel
            glon,glat = pixToAng(nside, pix)
            zval = zvalues[zidx]
            
            objs[i]['LABEL'] = good[i]
            objs[i]['NPIX'] = npix
            logger.debug("LABEL=%i"%objs[i]['LABEL'])
            logger.debug("NPIX=%i"%objs[i]['NPIX'])
     
            island = values[idx,zidx]
            idxmax = island.argmax()
            glon_max,glat_max,zval_max = glon[idxmax],glat[idxmax],zval[idxmax]
     
            objs[i]['VALUE_MAX'] = island[idxmax]
            objs[i]['IDX_MAX']  = idx[idxmax]
            objs[i]['ZIDX_MAX']  = zidx[idxmax]
            objs[i]['PIX_MAX']   = pix[idxmax]
            objs[i]['GLON_MAX']  = glon_max
            objs[i]['GLAT_MAX']  = glat_max
            objs[i]['ZVAL_MAX']  = zval_max
     
            proj = Projector(glon_max,glat_max)
            xpix,ypix = proj.sphereToImage(glon,glat)
            
            x_cent,y_cent,zval_cent = np.average([xpix,ypix,zval],axis=1)
            glon_cent, glat_cent = proj.imageToSphere(x_cent,y_cent)
            objs[i]['GLON_CENT'] = glon_cent
            objs[i]['GLAT_CENT'] = glat_cent
            objs[i]['ZVAL_CENT'] = zval_cent
     
            x_bary,y_bary,zval_bary = np.average([xpix,ypix,zval],weights=[island,island,island],axis=1)
            glon_bary, glat_bary = proj.imageToSphere(x_bary, y_bary)
            objs[i]['GLON_BARY'] = glon_bary
            objs[i]['GLAT_BARY'] = glat_bary
            objs[i]['ZVAL_BARY'] = zval_bary
     
        return objs

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()


    configfile= 'config_dr10_gal.py'
    config = ugali.utils.config.Config(configfile)


    
    nside = config.params['coords']['nside_pixel']
    threshold = 10
    xsize = 10000
    minpix = 1

    # Plotting
    #f = pyfits.open('draco_merged.fits')
    #f = pyfits.open('merged_all_v1.fits')
    f = pyfits.open('merged_v0.fits')
    #rot = [86.4,34.7]

    distances = f[2].data['DISTANCE_MODULUS']
    pixels,loglikes = f[1].data['pix'],f[1].data['log_likelihood']
    #values = 2*loglikes
    values = 2*loglikes

    labels, nlabels = labelHealpix(pixels, values, nside, threshold=threshold, xsize=xsize)
    hist,edges,rev = reverseHistogram(labels,bins=np.arange(nlabels+2))
    # Make some cut on the minimum size of a labelled object
    good, = np.where( (hist >= minpix) )
    # Get rid of zero label (below threshold)
    good, = np.nonzero(good)

    objs = createObjects(pixels, values, nside, distances, rev, good)

    """
    # Write labels
    data_dict = {'LOG_LIKELIHOOD':loglikes,'LABEL':labels}
    outfile="merged_labels_v1.fits"
    ugali.utils.skymap.writeSparseHealpixMap(pixels,data_dict,nside,outfile,
                                             distance_modulus_array=f[2].data['DISTANCE_MODULUS'],
                                             coordsys='NULL', ordering='NULL')
    """
    xsize = 1000
    radius = 5. #10.
    reso = radius*60/xsize

    map = healpy.UNSEEN * np.ones(healpy.nside2npix(nside))
    map[pixels] = labels.sum(axis=1)

    tsmap = healpy.UNSEEN * np.ones(healpy.nside2npix(nside))
    tsmap[pixels] = values.sum(axis=1)

    candidates = objs[objs['VALUE_MAX'] > 50]

    dwarfs    = ugali.association.sources.McConnachie12()
    globulars = ugali.association.sources.Harris96()
    clusters  = ugali.association.sources.Rykoff14()
    ngc       = ugali.association.sources.Steinicke10()

    unmatched = np.arange(len(candidates))
    dwarf_match = dwarfs.match(candidates['GLON_MAX'],candidates['GLAT_MAX'])
    unmatched = np.setdiff1d(unmatched,dwarf_match[0])
    # Increase radius for globular clusters due to SDSS masking
    globular_match = globulars.match(candidates['GLON_MAX'],candidates['GLAT_MAX'],tol=0.25)
    unmatched = np.setdiff1d(unmatched,globular_match[0])
    cluster_match = clusters.match(candidates['GLON_MAX'],candidates['GLAT_MAX'])
    unmatched = np.setdiff1d(unmatched,cluster_match[0])
    ngc_match = ngc.match(candidates['GLON_MAX'],candidates['GLAT_MAX'])
    unmatched = np.setdiff1d(unmatched,ngc_match[0])
     
    print "Matching Dwarf Galaxies..."
    for i,j,d in zip(*dwarf_match):
        source = dwarfs[j]
        print "\t",source['name'],source['ra'],source['dec'],d
     
    print "Matching Globular Clusters..."
    for i,j,d in zip(*globular_match):
        source = globulars[j]
        print "\t",source['name'],source['ra'],source['dec'],d
     
    print "Matching Galaxy Clusters..."
    for i,j,d in zip(*cluster_match):
        source = clusters[j]
        print "\t",source['name'],source['ra'],source['dec'],d

    print "Matching NGC Objects..."
    for i,j,d in zip(*ngc_match):
        source = ngc[j]
        print "\t",source['name'],source['ra'],source['dec'],d
     
    print "Unmatched objects..."
    for source in candidates[unmatched]:
        ra,dec = galToCel(source['GLON_MAX'],source['GLAT_MAX'])
        print "\t",source['VALUE_MAX'],ra,dec

    for obj in candidates:
        # Plotting
        #rot = [86.4,34.7] # Draco
        #rot = [353.7, 68.9] # Bootes I
        #rot = [obj['GLON_MAX'],obj['GLAT_MAX']]

        ra,dec = galToCel(obj['GLON_MAX'],obj['GLAT_MAX'])
        rot = [ra,dec]
        print rot
        fig,ax = plt.subplots(2,2,figsize=(8,8))
        plt.sca(ax[0][0])
        kwargs = dict(rot=[ra,dec],lonra=[-.5,.5],latra=[-.5,.5],xsize=xsize,hold=True,coord='GC',notext=True)
        healpy.cartview(map,**kwargs)
        healpy.graticule(dpar=1,dmer=1,color='0.5',verbose=False)
        #healpy.gnomview(map,rot=rot,reso=reso,xsize=xsize,hold=True,coord='GC')
        plt.sca(ax[0][1])
        healpy.cartview(tsmap,**kwargs)
        healpy.graticule(dpar=1,dmer=1,color='0.5',verbose=False)
        #healpy.gnomview(tsmap,rot=rot,reso=reso,xsize=xsize,hold=True,coord='GC')
        plt.sca(ax[1][0])
        healpy.cartview(magmap,**kwargs)
        healpy.graticule(dpar=1,dmer=1,color='0.5',verbose=False)

        plt.sca(ax[1][1])
        url="http://skyservice.pha.jhu.edu/DR9/ImgCutout/getjpeg.aspx?"
        params=dict(ra=ra,dec=dec,
                    width=1600,height=1600,
                    scale=2.0,opt='GMPL')
        query='&'.join("%s=%s"%(k,v) for k,v in params.items())
        outfile="tmp_%(ra).3f_%(dec).3f.jpeg"%params
        cmd='wget -O %s "%s"'%(outfile,url+query)
        subprocess.call(cmd,shell=True)
        im = plt.imread(outfile)
        plt.gca().imshow(im,origin='upper')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        os.remove(outfile)

    plt.ion()

