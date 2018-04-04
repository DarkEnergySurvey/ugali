#!/usr/bin/env python
"""
For creating healpix maps from catalogs.
"""
import os
from os.path import join
import shutil

import pyfits
import fitsio
import numpy
import numpy as np

import numpy.lib.recfunctions as recfuncs
import tempfile
import subprocess
import healpy
from collections import Counter
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import ugali.utils.skymap
import ugali.utils.binning
from ugali.utils.projector import cel2gal, gal2cel
from ugali.utils.healpix import ang2pix, pix2ang,superpixel
from ugali.utils.shell import mkdir
from ugali.utils.logger import logger
from ugali.utils.config import Config
from ugali.utils.constants import MAGLIMS


class Maglims(object):
    def __init__(self, config):
        self.config = Config(config)
        self._setup()

    def _setup(self):
        self.nside_catalog = self.config['coords']['nside_catalog']
        self.nside_mask = self.config['coords']['nside_mask']
        self.nside_pixel = self.config['coords']['nside_pixel']
        
        self.filenames = self.config.getFilenames()
        
        self.footfile = self.config['data']['footprint']
        try: 
            self.footprint = fitsio.read(self.footfile)['I'].ravel()
        except:
            logger.warn("Couldn't open %s; will try again."%self.footfile)
            self.footprint = self.footfile


    def run(self,field=None,simple=False,force=False):
        """
        Loop through pixels containing catalog objects and calculate
        the magnitude limit. This gets a bit convoluted due to all
        the different pixel resolutions...
        """
        if field is None: fields = [1,2]
        else:             fields = [field]
        for filenames in self.filenames.compress(~self.filenames.mask['catalog']).data:
            infile = filenames['catalog']
            for f in fields:
                outfile = filenames['mask_%i'%f]
                if os.path.exists(outfile) and not force:
                    logger.info("Found %s; skipping..."%outfile)
                    continue
                
                pixels,maglims=self.calculate(infile,f,simple)
                logger.info("Creating %s"%outfile)
                outdir = mkdir(os.path.dirname(outfile))
                #data_dict = dict( MAGLIM=maglims )
                #ugali.utils.skymap.writeSparseHealpixMap(pixels,data_dict,self.nside_pixel,outfile)
                data = dict(PIXEL=pixels,MAGLIM=maglims )
                ugali.utils.healpix.write_partial_map(outfile,data,self.nside_pixel,
                                                      dtype='f4')

    def calculate(self, infile, field=1, simple=False):
        logger.info("Calculating magnitude limit from %s"%infile)

        #manglefile = self.config['mangle']['infile_%i'%field]
        #footfile = self.config['data']['footprint']
        #try: 
        #    footprint = fitsio.read(footfile)['I'].ravel()
        #except:
        #    logger.warn("Couldn't open %s; will try again."%footfile)
        #    footprint = footfile

        mag_column = self.config['catalog']['mag_%i_field'%field]
        magerr_column = self.config['catalog']['mag_err_%i_field'%field]

        # For simple maglims
        release = self.config['data']['release'].lower()
        band    = self.config['catalog']['mag_%i_band'%field]
        pixel_pix_name = 'PIX%i'%self.nside_pixel         

        data = fitsio.read(infile,columns=[pixel_pix_name])

        #mask_pixels = numpy.arange( healpy.nside2npix(self.nside_mask), dtype='int')
        mask_maglims = numpy.zeros( healpy.nside2npix(self.nside_mask) )
         
        out_pixels = numpy.zeros(0,dtype='int')
        out_maglims = numpy.zeros(0)
         
        # Find the objects in each pixel
        pixel_pix = data[pixel_pix_name]
        mask_pix = ugali.utils.skymap.superpixel(pixel_pix,self.nside_pixel,self.nside_mask)
        count = Counter(mask_pix)
        pixels = sorted(count.keys())
        pix_digi = numpy.digitize(mask_pix,pixels).argsort()
        idx = 0
        min_num = 500
        signal_to_noise = 10.
        magerr_lim = 1/signal_to_noise
        for pix in pixels:
            # Calculate the magnitude limit in each pixel
            num = count[pix]
            objs = data[pix_digi[idx:idx+num]]
            idx += num
            if simple:
                # Set constant magnitude limits
                logger.debug("Simple magnitude limit for %s"%infile)
                mask_maglims[pix] = MAGLIMS[release][band]
            elif num < min_num:
                logger.info('Found <%i objects in pixel %i'%(min_num,pix))
                mask_maglims[pix] = 0
            else:
                mag = objs[mag_column]
                magerr = objs[magerr_column]
                # Estimate the magnitude limit as suggested by:
                # https://deswiki.cosmology.illinois.edu/confluence/display/DO/SVA1+Release+Document
                # (https://desweb.cosmology.illinois.edu/confluence/display/Operations/SVA1+Doc)
                maglim = numpy.median(mag[(magerr>0.9*magerr_lim)&(magerr<1.1*magerr_lim)])
         
                # Alternative method to estimate the magnitude limit by fitting median
                #mag_min, mag_max = mag.min(),mag.max()
                #mag_bins = numpy.arange(mag_min,mag_max,0.1) #0.1086?
                #x,y = ugali.utils.binning.binnedMedian(mag,magerr,mag_bins)
                #x,y = x[~numpy.isnan(y)],y[~numpy.isnan(y)]
                #magerr_med = interp1d(x,y)
                #mag0 = numpy.median(x) 
                #maglim = brentq(lambda a: magerr_med(a)-magerr_lim,x.min(),x.max(),disp=False)
                # Median from just objects near magerr cut
         
                mask_maglims[pix] = maglim

            logger.debug("%i (n=%i): maglim=%g"%(pix,num,mask_maglims[pix]))
            subpix = ugali.utils.skymap.subpixel(pix, self.nside_mask, self.nside_pixel)
            maglims = numpy.zeros(len(subpix)) + mask_maglims[pix] 
            out_pixels = numpy.append(out_pixels,subpix)
            out_maglims = numpy.append(out_maglims,maglims)
         
        # Remove empty pixels
        logger.info("Removing empty pixels")
        idx = numpy.nonzero(out_maglims > 0)[0]
        out_pixels  = out_pixels[idx]
        out_maglims = out_maglims[idx]
         
        # Remove pixels outside the footprint
        logger.info("Checking footprint against %s"%self.footfile)
        glon,glat = pix2ang(self.nside_pixel,out_pixels)
        ra,dec = gal2cel(glon,glat)
        footprint = inFootprint(self.footprint,ra,dec)
        idx = numpy.nonzero(footprint)[0]
        out_pixels = out_pixels[idx]
        out_maglims = out_maglims[idx]
         
        logger.info("MAGLIM = %.3f +/- %.3f"%(numpy.mean(out_maglims),numpy.std(out_maglims)))         
        return out_pixels,out_maglims

def inFootprint(footprint,ra,dec):
    """
    Check if set of ra,dec combinations are in footprint.
    Careful, input files must be in celestial coordinates.
    
    filename : Either healpix map or mangle polygon file
    ra,dec   : Celestial coordinates

    Returns:
    inside   : boolean array of coordinates in footprint
    """
        
    try:
        if isinstance(footprint,str) and os.path.exists(footprint):
            filename = footprint
            #footprint = healpy.read_map(filename,verbose=False)
            footprint = fitsio.read(filename)['I'].ravel()
        nside = healpy.npix2nside(len(footprint))
        pix = ang2pix(nside,ra,dec)
        inside = (footprint[pix] > 0)
    except IOError:
        logger.warning("Failed to load healpix footprint; trying to use mangle...")
        inside = inMangle(filename,ra,dec)
    return inside

def inMangle(polyfile,ra,dec):
    coords = tempfile.NamedTemporaryFile(suffix='.txt',delete=False)
    logger.debug("Writing coordinates to %s"%coords.name)
    numpy.savetxt(coords, numpy.array( [ra,dec] ).T, fmt='%.6g' )
    coords.close()

    weights = tempfile.NamedTemporaryFile(suffix='.txt',delete=False)
    cmd = "polyid -W %s %s %s"%(polyfile,coords.name,weights.name)
    logger.debug(cmd)
    subprocess.call(cmd,shell=True)

    tmp = tempfile.NamedTemporaryFile(suffix='.txt',delete=False)
    cmd = """awk '{if($3==""){$3=0} print $1, $2, $3}' %s > %s"""%(weights.name,tmp.name)
    logger.debug(cmd)
    subprocess.call(cmd,shell=True)

    data = numpy.loadtxt(tmp.name,unpack=True,skiprows=1)[-1]
    for f in [coords,weights,tmp]:
        logger.debug("Removing %s"%f.name)
        os.remove(f.name)

    return data > 0

def simple_maglims(config,dirname='simple',force=False):
    """
    Create simple, uniform magnitude limits based on nominal
    survey depth.
    """
    filenames = config.getFilenames()
    release = config['data']['release'].lower()
    #band_1 = config['isochrone']['mag_1_field']
    #band_2 = config['isochrone']['mag_2_field']
    band_1 = config['catalog']['mag_1_field']
    band_2 = config['catalog']['mag_2_field']
    mask_1 = filenames['mask_1'].compressed()
    mask_2 = filenames['mask_2'].compressed()
    basedir,basename = os.path.split(config['mask']['dirname'])
    if basename == dirname:
        raise Exception("Input and output directory are the same.")
    outdir = mkdir(os.path.join(basedir,dirname))

    for band, infiles in [(band_1,mask_1),(band_2,mask_2)]:
        maglim = MAGLIMS[release][band]
        for infile in infiles:
            basename = os.path.basename(infile)
            outfile = join(outdir,basename)
            logger.debug('Reading %s...'%infile)
            f = pyfits.open(infile)
            f[1].data['MAGLIM'][:] = maglim
            logger.debug('Writing %s...'%outfile)
            f.writeto(outfile,clobber=True)

def simple_split(config,dirname='split',force=False):
    config = Config(config)
    filenames = config.getFilenames()
    healpix = filenames['pix'].compressed()

    nside_catalog = config['coords']['nside_catalog']
    nside_pixel = config['coords']['nside_pixel']

    release = config['data']['release'].lower()
    band_1 = config['catalog']['mag_1_band']
    band_2 = config['catalog']['mag_2_band']

    mangledir = config['mangle']['dirname']

    mangle_file_1 = join(mangledir,config['mangle']['filename_1'])
    logger.info("Reading %s..."%mangle_file_1)
    mangle_1 = healpy.read_map(mangle_file_1)
    
    mangle_file_2 = join(mangledir,config['mangle']['filename_2'])
    logger.info("Reading %s..."%mangle_file_2)
    mangle_2 = healpy.read_map(mangle_file_2)

    basedir,basename = os.path.split(config['mask']['dirname'])
    if basename == dirname:
        msg = "Input and output directory are the same."
        raise Exception(msg)
    outdir = mkdir(os.path.join(basedir,dirname))

    mask_1 = os.path.basename(config['mask']['basename_1'])
    mask_2 = os.path.basename(config['mask']['basename_2'])

    for band,mangle,base in [(band_1,mangle_1,mask_1),(band_2,mangle_2,mask_2)]:
        maglim = MAGLIMS[release][band]

        nside_mangle = healpy.npix2nside(len(mangle))
        if nside_mangle != nside_pixel:
            msg = "Mangle nside different from pixel nside"
            logger.warning(msg)
            #raise Exception(msg)


        pixels = np.nonzero((mangle>0)&(mangle>maglim))[0]
        print len(pixels)
        superpix = superpixel(pixels,nside_mangle,nside_catalog)
        print healpix
        for hpx in healpix:
            outfile = join(outdir,base)%hpx
            if os.path.exists(outfile) and not force:
                logger.warning("Found %s; skipping..."%outfile)
                continue

            pix = pixels[superpix == hpx]
            print hpx, len(pix)

            maglims = maglim*np.ones(len(pix))
            logger.info('Writing %s...'%outfile)
            #data = dict(MAGLIM=maglims )
            #ugali.utils.skymap.writeSparseHealpixMap(pix,data,nside_pixel,outfile)
            data = dict(PIXEL=pix,MAGLIM=maglims)
            ugali.utils.healpix.write_partial_map(outfile,data,nside_pixel,dtype='f4')
        
if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()
