#!/usr/bin/env python
"""
For creating healpix maps from catalogs.
"""
import os
from os.path import join
import shutil

import pyfits
import numpy
import numpy.lib.recfunctions as recfuncs
import tempfile
import subprocess
import healpy
from collections import Counter
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import ugali.utils.skymap
import ugali.utils.binning
from ugali.utils.projector import celToGal, galToCel
from ugali.utils.projector import angToPix, pixToAng
from ugali.utils.shell import mkdir
from ugali.utils.logger import logger
from ugali.utils.config import Config

# http://www.adsabs.harvard.edu/abs/2002AJ....123..485S
# Determination of magnitude limits is rather complicated.
# The technique applied here is to derive the magnitude at
# which the 10sigma signal-to-noise threshold is reached.
# For SDSS, these values are (Table 21):
# u=22.12,g=22.60,r=22.29,i=21.85,z=20.35 
# However, the quoted 95% completeness limits are (Table 2):
# u=22.0,g=22.2,r=22.2,i=21.3,z=20.5
# What is responsible for this disconnect? Well, I think 
# the completeness was estimated by comparing with COSMOS
# on nights that weren't all that good seeing.

# http://des-docdb.fnal.gov:8080/cgi-bin/ShowDocument?docid=20
# DES simple magnitude limits come from the science 
# requirements document. These are 'requirements'
# are somewhat pessimistic and the document also
# conatains 'goals' for the magnitude limit:
# g=25.4, r=24.9, i=24.6, z=23.8, y=21.7
# Of course all of this is yet to be verified with
# data...

SIMPLE_MAGLIMS = dict(
    sdss = {
        'u': 22.12,
        'g': 22.60,
        'r': 22.29,
        'i': 22.85,
        'z': 20.35
    },
    des = {
        'g': 24.6,
        'r': 24.1,
        'i': 24.3,
        'z': 23.8,
        'Y': 21.5
    }
)


class Maglims(object):
    def __init__(self, config):
        self.config = config
        self._setup()

    def _setup(self):
        self.nside_catalog = self.config.params['coords']['nside_catalog']
        self.nside_mask = self.config.params['coords']['nside_mask']
        self.nside_pixel = self.config.params['coords']['nside_pixel']
        
        self.filenames = self.config.getFilenames()

    def run(self,field=None):
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
                self.calculate(infile,outfile,f)
    
    def calculate(self, infile, outfile, field=1):
        logger.info("Calculating magnitude limit from %s"%infile)

        manglefile = self.config.params['mangle']['infile_%i'%field]
        mag_column = self.config.params['catalog']['mag_%i_field'%field]
        magerr_column = self.config.params['catalog']['mag_err_%i_field'%field]
         
        f = pyfits.open(infile)
        header = f[1].header
        data = f[1].data
         
        mask_pixels = numpy.arange( healpy.nside2npix(self.nside_mask), dtype='int')
        mask_maglims = numpy.zeros( healpy.nside2npix(self.nside_mask) )
         
        out_pixels = numpy.zeros(0,dtype='int')
        out_maglims = numpy.zeros(0)
         
        # Find the objects in each pixel
        pixel_pix = data['PIX%i'%self.nside_pixel]
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
            if num < min_num:
                logger.info('Found <%i objects in pixel %i'%(min_num,pix))
                mask_maglims[pix] = 0
            else:
                mag = objs[mag_column]
                magerr = objs[magerr_column]
                # Estimate the magnitude limit as suggested by:
                # https://desweb.cosmology.illinois.edu/confluence/display/Operations/SVA1+Doc
                maglim = numpy.median(mag[(magerr>0.9*magerr_lim)&(magerr<1.1*magerr_lim)])
         
                # Alternative method to estimate the magnitude limit by fitting median
                #mag_min, mag_max = mag.min(),mag.max()
                #mag_bins = numpy.arange(mag_min,mag_max,0.1)
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
        logger.info("Checking footprint against %s"%manglefile)
        glon,glat = pixToAng(self.nside_pixel,out_pixels)
        ra,dec = galToCel(glon,glat)
        footprint = inMangle(manglefile,ra,dec)
        idx = numpy.nonzero(footprint)[0]
        out_pixels = out_pixels[idx]
        out_maglims = out_maglims[idx]
         
        logger.info("MAGLIM = %.3f +/- %.3f"%(numpy.mean(out_maglims),numpy.std(out_maglims)))
         
        logger.info("Creating %s"%outfile)
        outdir = mkdir(os.path.dirname(outfile))

        data_dict = dict( MAGLIM=out_maglims )
        ugali.utils.skymap.writeSparseHealpixMap(out_pixels,data_dict,self.nside_pixel,outfile)
         
        return out_pixels,out_maglims

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

def simple_maglims(config,dirname='simple'):
    """
    Creat simple, uniform magnitude limits based on nominal
    survey depth.
    """
    filenames = config.getFilenames()
    survey = config.params['data']['survey'].lower()
    band_1 = config.params['isochrone']['mag_1_field']
    band_2 = config.params['isochrone']['mag_2_field']
    mask_1 = filenames['mask_1'].compressed()
    mask_2 = filenames['mask_2'].compressed()
    basedir,basename = os.path.split(config.params['mask']['dirname'])
    if basename == dirname:
        raise Exception("Input and output directory are the same.")
    outdir = mkdir(os.path.join(basedir,dirname))

    for band, infiles in [(band_1,mask_1),(band_2,mask_2)]:
        maglim = SIMPLE_MAGLIMS[survey][band]
        for infile in infiles:
            basename = os.path.basename(infile)
            outfile = join(outdir,basename)
            logger.debug('Reading %s...'%infile)
            f = pyfits.open(infile)
            f[1].data['MAGLIM'][:] = maglim
            logger.debug('Writing %s...'%outfile)
            f.writeto(outfile,clobber=True)


if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()
