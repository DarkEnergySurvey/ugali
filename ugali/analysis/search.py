#!/usr/bin/env python
"""
@author: Alex Drlica-Wagner <kadrlica@fnal.gov>
"""

import os
import copy
import subprocess
from collections import OrderedDict as odict

import healpy
import pyfits
import fitsio
import numpy as np
import numpy
import numpy.lib.recfunctions as recfuncs
import scipy.ndimage as ndimage

import ugali.candidate.associate
import ugali.utils.skymap

from ugali.utils.shell import mkdir, which
from ugali.utils.logger import logger
from ugali.utils.binning import reverseHistogram
from ugali.utils.projector import Projector, gal2cel, dec2hms, dec2dms, mod2dist
from ugali.utils.healpix import pix2ang, ang2pix

class CandidateSearch(object):
    """
    Object used to search for candidate objects in TS maps.
    """
    def __init__(self, config, mergefile=None, roifile=None):
        self.config = config
        self._config()
        if mergefile is not None: self.mergefile = mergefile
        if roifile is not None: self.roifile = roifile
        self._load()

    def _config(self):
        self.nside = self.config['coords']['nside_pixel']
        self.threshold = self.config['search']['obj_threshold'] # = 10
        self.xsize = self.config['search']['xsize'] # = 10000
        self.minpix = self.config['search']['minpix'] # = 1

        self.mergefile  = self.config.mergefile
        self.roifile    = self.config.roifile
        self.labelfile  = self.config.labelfile
        self.objectfile = self.config.objectfile
        self.assocfile  = self.config.assocfile
        self.candfile   = self.config.candfile

        mkdir(self.config['output']['searchdir'])
              
    def _load(self):
        self.loadLikelihood()
        self.loadROI()

    def loadLikelihood(self,filename=None):
        if filename is None: filename = self.mergefile
        #f = pyfits.open(self.mergefile)
        #self.pixels = f[1].data['PIX']
        #self.values = 2*f[1].data['LOG_LIKELIHOOD']
        #self.distances = f[2].data['DISTANCE_MODULUS']
        #self.richness = f[1].data['RICHNESS']
        f = fitsio.FITS(self.mergefile)
        data = f[1].read()
        self.pixels = data['PIXEL'] if 'PIXEL' in data.dtype.names else data['PIX']
        self.values = 2*data['LOG_LIKELIHOOD']
        self.distances = f[2].read()['DISTANCE_MODULUS']
        self.richness = data['RICHNESS']

    def loadROI(self,filename=None):
        if filename is None: filename = self.roifile
        self.ninterior = ugali.utils.skymap.readSparseHealpixMap(filename,'NINSIDE')
        self.nannulus = ugali.utils.skymap.readSparseHealpixMap(filename,'NANNULUS')
        self.stellar = ugali.utils.skymap.readSparseHealpixMap(filename,'STELLAR')

    def createLabels2D(self):
        """ 2D labeling at zmax """
        logger.debug("  Creating 2D labels...")
        self.zmax = np.argmax(self.values,axis=1)
        self.vmax = self.values[np.arange(len(self.pixels),dtype=int),self.zmax]

        kwargs=dict(pixels=self.pixels,values=self.vmax,nside=self.nside,
                    threshold=self.threshold,xsize=self.xsize)
        labels,nlabels = CandidateSearch.labelHealpix(**kwargs)
        self.nlabels = nlabels
        self.labels = np.repeat(labels,len(self.distances)).reshape(len(labels),len(self.distances))
        return self.labels, self.nlabels

    def createLabels3D(self):
        logger.debug("  Creating 3D labels...")
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
        #f = pyfits.open(filename)
        data = fitsio.read(filename)
        if not (self.pixels == data['PIXEL']).all(): 
            raise Exception("...")
        if not (self.distances == data['DISTANCE_MODULUS']).all():
            raise Exception("...")            

        self.labels = data['LABEL'].astype(int)
        self.nlabels = self.labels.max()
        if self.nlabels != (len(np.unique(self.labels)) - 1):
            raise Exception("Incorrect number of labels found.")
        return self.labels, self.nlabels
        
    def createObjects(self):
        logger.debug("  Creating objects...")
        hist,edges,rev = reverseHistogram(self.labels,bins=numpy.arange(self.nlabels+2))
        self.rev = rev
        # Make some cut on the minimum size of a labelled object
        good, = numpy.where( (hist >= self.minpix) )
        # Get rid of zero label (below threshold)
        self.good, = numpy.nonzero(good)

        kwargs=dict(pixels=self.pixels,values=self.values,nside=self.nside, 
                    zvalues=self.distances, rev=self.rev, good=self.good)
        objects = self.findObjects(**kwargs)
        self.objects = self.finalizeObjects(objects)

        return self.objects

    def writeObjects(self,filename=None):
        if filename is None: filename = self.objectfile
        #hdu = pyfits.new_table(self.objects.view(np.recarray))
        #logger.info("Writing %s..."%filename)
        #hdu.writeto(filename,clobber=True)
        logger.info("Writing %s..."%filename)
        fitsio.write(filename,self.objects.view(np.recarray),clobber=True)


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
            logger.debug("Labeling slice %i...")
            searchim = numpy.zeros(xx.shape,dtype=bool)
            select = (value > threshold)
            yidx = ij[0][select]; xidx = ij[1][select]
            searchim[yidx,xidx] |= True
            searchims.append( searchim )
        searchims = numpy.array(searchims)

        # Full binary structure
        s = ndimage.generate_binary_structure(searchims.ndim,searchims.ndim)
     
        ### # Dilate in the z-direction
        logger.info("  Dilating image...")
        searchims = ndimage.binary_dilation(searchims,s,1)
        
        # Do the labeling
        logger.info("  Labeling image...")
        labels,nlabels = ndimage.label(searchims,structure=s)

        # Convert back to healpix
        pix_labels = labels[:,ij[0],ij[1]].T
        pix_labels = pix_labels.reshape(values.shape)
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
                                  ('VAL_MAX','f4'),
                                  ('IDX_MAX','i4'),
                                  ('ZIDX_MAX','i4'),
                                  ('PIX_MAX','i4'),
                                  ('X_MAX','f4'),
                                  ('Y_MAX','f4'),
                                  ('Z_MAX','f4'),
                                  ('X_CENT','f4'),
                                  ('Y_CENT','f4'),
                                  ('Z_CENT','f4'),
                                  ('X_BARY','f4'),
                                  ('Y_BARY','f4'),
                                  ('Z_BARY','f4'),
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
            xval,yval = pix2ang(nside, pix)
            zval = zvalues[zidx]
            
            objs[i]['LABEL'] = good[i]
            objs[i]['NPIX'] = npix
            logger.debug("LABEL=%i"%objs[i]['LABEL'])
            logger.debug("NPIX=%i"%objs[i]['NPIX'])
     
            island = values[idx,zidx]
            idxmax = island.argmax()
            xval_max,yval_max,zval_max = xval[idxmax],yval[idxmax],zval[idxmax]
     
            objs[i]['VAL_MAX'] = island[idxmax]
            objs[i]['IDX_MAX']  = idx[idxmax]
            objs[i]['ZIDX_MAX']  = zidx[idxmax]
            objs[i]['PIX_MAX']   = pix[idxmax]
            objs[i]['X_MAX']  = xval_max
            objs[i]['Y_MAX']  = yval_max
            objs[i]['Z_MAX']  = zval_max

            proj = Projector(xval_max,yval_max)
            xpix,ypix = proj.sphereToImage(xval,yval)

            # Projected centroid
            x_cent,y_cent,zval_cent = numpy.average([xpix,ypix,zval],axis=1)
            xval_cent, yval_cent = proj.imageToSphere(x_cent,y_cent)
            objs[i]['X_CENT'] = xval_cent
            objs[i]['Y_CENT'] = yval_cent
            objs[i]['Z_CENT'] = zval_cent

            # Projected barycenter
            weights=[island,island,island]
            x_bary,y_bary,zval_bary = numpy.average([xpix,ypix,zval],weights=weights,axis=1)
            xval_bary,yval_bary = proj.imageToSphere(x_bary, y_bary)
            objs[i]['X_BARY'] = xval_bary
            objs[i]['Y_BARY'] = yval_bary
            objs[i]['Z_BARY'] = zval_bary
     
        return objs

    def finalizeObjects(self, objects):
        objs = numpy.recarray(len(objects),
                              dtype=[('NAME','S24'),
                                     ('TS','f4'),
                                     ('GLON','f4'),
                                     ('GLAT','f4'),
                                     ('RA','f4'),
                                     ('DEC','f4'),
                                     ('MODULUS','f4'),
                                     ('DISTANCE','f4'),
                                     ('RICHNESS','f4'),
                                     ('MASS','f4'),
                                     ('NANNULUS','i4'),
                                     ('NINTERIOR','i4'),
                                     ])
        
        objs['TS'] = self.values[objects['IDX_MAX'],objects['ZIDX_MAX']]

        glon,glat = objects['X_MAX'],objects['Y_MAX']
        objs['GLON'],objs['GLAT'] = glon,glat
        
        ra,dec    = gal2cel(glon,glat)
        objs['RA'],objs['DEC'] = ra,dec

        modulus = objects['Z_MAX']
        objs['MODULUS'] = modulus
        objs['DISTANCE'] = mod2dist(modulus)

        #ninterior = ugali.utils.skymap.readSparseHealpixMap(self.roifile,'NINSIDE')
        #nannulus = ugali.utils.skymap.readSparseHealpixMap(self.roifile,'NANNULUS')
        #stellar = ugali.utils.skymap.readSparseHealpixMap(self.roifile,'STELLAR')

        nside = healpy.npix2nside(len(self.nannulus))
        pix = ang2pix(nside,glon,glat)

        richness = self.richness[objects['IDX_MAX'],objects['ZIDX_MAX']]
        objs['RICHNESS'] = richness
        objs['MASS'] = richness * self.stellar[pix]

        objs['NANNULUS']  = self.nannulus[pix].astype(int)
        objs['NINTERIOR'] = self.ninterior[pix].astype(int)

        # Default name formatting
        # http://cdsarc.u-strasbg.fr/ftp/pub/iau/
        # http://cds.u-strasbg.fr/vizier/Dic/iau-spec.htx
        fmt = "J%(hour)02i%(hmin)04.1f%(deg)+03i%(dmin)02i"
        for obj,_ra,_dec in zip(objs,ra,dec):
            hms = dec2hms(_ra); dms = dec2dms(_dec)
            params = dict(hour=hms[0],hmin=hms[1]+hms[2]/60.,
                          deg=dms[0],dmin=dms[1]+dms[2]/60.)
            obj['NAME'] = fmt%params

        out = recfuncs.merge_arrays([objs,objects],usemask=False,asrecarray=True,flatten=True)
        # This is safer than viewing as FITS_rec
        return pyfits.new_table(out).data

    def loadLabels(self,filename=None):
        if filename is None: filename = self.labelfile
        f = pyfits.open(filename)
        if not (self.pixels == f[1].data['PIX']).all(): 
            raise Exception("...")
        if not (self.distances == f[2].data['DISTANCE_MODULUS']).all():
            raise Exception("...")            

        self.labels = f[1].data['LABEL'].astype(int)
        self.nlabels = self.labels.max()
        if self.nlabels != (len(np.unique(self.labels)) - 1):
            raise Exception("Incorrect number of labels found.")
        return self.labels, self.nlabels

    def loadObjects(self,filename=None):
        if filename is None: filename = self.objectfile
        f = pyfits.open(filename)
        self.objects = f[1].data

    def loadAssociations(self,filename=None):
        if filename is None: filename = self.assocfile
        f = pyfits.open(filename)
        self.assocs = f[1].data

    def createAssociations(self):
        objects = self.objects

        tol = self.config['search']['proximity']
        columns = []
     
        names = np.empty(len(objects),dtype=object)
        names.fill('')
        for i,refs in enumerate(self.config['search']['catalogs']):
            i += 1
            catalog = ugali.candidate.associate.SourceCatalog()
            for ref in refs:
                catalog += ugali.candidate.associate.catalogFactory(ref)
     
            # String length (should be greater than longest name)
            length = len(max(catalog['name'],key=len)) + 1
            dtype = 'S%i'%length; fitstype='%iA'%length
     
            assoc = np.empty(len(objects),dtype=dtype)
            assoc.fill('')
            idx1,idx2,dist = catalog.match(objects['GLON'],objects['GLAT'],tol=tol)
            assoc[idx1] = catalog['name'][idx2].astype(dtype)
            columns.append(pyfits.Column(name='ASSOC%i'%i,format=fitstype,array=assoc))
            columns.append(pyfits.Column(name='ANGSEP%i'%i,format='E',array=dist))

            if length > objects['NAME'].itemsize:
                logger.warning("Association name may not fit.")
            names = np.where(names=='',assoc,names)
        names = names.astype(objects['NAME'].dtype)
        objects['NAME'][:] = np.where(names=='',objects['NAME'],names)

        self.assocs = pyfits.new_table(objects.columns+pyfits.ColDefs(columns)).data
        self.assocs = self.assocs[self.assocs['NAME'].argsort()]

    def writeAssociations(self,filename=None):
        if filename is None: filename = self.assocfile
        hdu = pyfits.new_table(self.assocs.view(np.recarray))
        logger.info("Writing %s..."%filename)
        hdu.writeto(filename,clobber=True)

    def writeCandidates(self,filename=None):
        if filename is None: filename = self.candfile

        threshold = self.config['search']['cand_threshold']
        select  = (self.assocs['CUT']==0)
        select &= (self.assocs['TS']>threshold)
        #select &= (self.assocs['ASSOC2']=='')

        self.candidates = self.assocs[select]
        # ADW: View as a recarray or selection doesn't work.
        # Why? I don't know, and I'm slightly terrified...
        hdu = pyfits.new_table(self.candidates.view(np.recarray))
        logger.info("Writing %s..."%filename)
        hdu.writeto(filename,clobber=True)

        # DEPRECATED: ADW 2017-09-15 
        ## Dump to txt file 
        #if which('fdump'):
        #    txtfile = filename.replace('.fits','.txt')
        #    columns = ['NAME','TS','GLON','GLAT','DISTANCE','MASS']
        #    cmd = 'fdump %(infile)s %(outfile)s columns="%(columns)s" rows="-" prhead="no" showcol="yes" clobber="yes" pagewidth="256" fldsep=" " showrow="no"'%(dict(infile=filename,outfile=txtfile,columns=','.join(columns)))
        #    print cmd
        #    subprocess.call(cmd,shell=True)

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()
