#!/usr/bin/env python
import healpy
import pyfits
import numpy as np
import numpy
import pylab as plt
import scipy.ndimage as ndimage
import copy
import subprocess
import os

import ugali.utils.skymap
import ugali.utils.parse_config
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
            dirname = config.params['output']['savedir_results']
            basename = config.params['output']['mergefile']
            filename = os.path.join(dirname,basename)
        self.filename = filename
        self._load()
        self._config()

    def _config(self):
        self.nside = self.config.params['coords']['nside_pixel']
        self.threshold = self.config.params['search']['threshold'] # = 10
        self.xsize = self.config.params['search']['xsize'] # = 10000
        self.minpix = self.config.params['search']['minpix'] # = 1
        dirname=self.config.params['output']['savedir_results']
        basename=self.config.params['output']['labelfile']
        self.labelfile = os.path.join(dirname,basename)
        basename=self.config.params['output']['objectfile']
        self.objectfile = os.path.join(dirname,basename)
        
    def _load(self):
        f = pyfits.open(self.filename)
        self.pixels = f[1].data['pix']
        self.values = 2*f[1].data['log_likelihood']
        self.distances = f[2].data['DISTANCE_MODULUS']

    def createLabels(self):
        logger.debug("Creating labels...")
        kwargs=dict(pixels=self.pixels,values=self.values,nside=self.nside,
                    threshold=self.threshold,xsize=self.xsize)
        self.labels,self.nlabels = CandidateSearch.labelHealpix(**kwargs)
        return self.labels, self.nlabels

    def writeLabels(self,filename=None):
        if filename is None: filename = self.labelfile
        # Converting to float is a waste of memory...
        # This should be much more robustly done in writeSparseHealpixMap
        data_dict = {'LABEL':self.labels.astype(float)}
        logger.info("Writing %s..."%filename)
        ugali.utils.skymap.writeSparseHealpixMap(self.pixels,data_dict,self.nside,filename,
                                                 distance_modulus_array=self.distances)

    def loadLabels(self,filename=None):
        if filename is None: filename = self.labelfile
        f = pyfits.open(filename)
        if not (self.pixels == f[1].data['pix']).all(): 
            raise Exception("...")
        if not (self.distances == f[2].data['DISTANCE_MODULUS']).all():
            raise Exception("...")            

        self.labels = f[1].data['label'].astype(int)
        self.nlabels = self.labels.max()
        if self.nlabels != (len(np.unique(self.labels)) - 1):
            raise Exception("Incorrect number of labels found.")
        return self.labels, self.nlabels
        
    def createObjects(self):
        logger.info("Creating objects...")
        hist,edges,rev = reverseHistogram(self.labels,bins=numpy.arange(self.nlabels+2))
        self.rev = rev
        # Make some cut on the minimum size of a labelled object
        good, = numpy.where( (hist >= self.minpix) )
        # Get rid of zero label (below threshold)
        self.good, = numpy.nonzero(good)

        kwargs=dict(pixels=self.pixels,values=self.values,nside=self.nside, 
                    zvalues=self.distances, rev=self.rev, good=self.good)

        self.objects = CandidateSearch.findObjects(**kwargs)
        return self.objects

    def writeObjects(self,filename=None):
        if filename is None: filename = self.objectfile
        hdu = pyfits.new_table(self.objects)
        logger.info("Writing %s..."%filename)
        hdu.writeto(filename,clobber=True)

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
            searchim = numpy.zeros(xx.shape,dtype=bool)
            select = (value > threshold)
            yidx = ij[0][select]; xidx = ij[1][select]
            searchim[yidx,xidx] |= True
            searchims.append( searchim )
     
        # Do the labeling
        searchims = numpy.array(searchims)
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
        objs = numpy.recarray((ngood,),
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
            
            x_cent,y_cent,zval_cent = numpy.average([xpix,ypix,zval],axis=1)
            glon_cent, glat_cent = proj.imageToSphere(x_cent,y_cent)
            objs[i]['GLON_CENT'] = glon_cent
            objs[i]['GLAT_CENT'] = glat_cent
            objs[i]['ZVAL_CENT'] = zval_cent
     
            x_bary,y_bary,zval_bary = numpy.average([xpix,ypix,zval],weights=[island,island,island],axis=1)
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

    
