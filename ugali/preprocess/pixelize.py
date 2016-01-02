#!/usr/bin/env python
"""
For pixelizing catalogs and masks.
"""
import os
from os.path import join

import pyfits
import numpy
import numpy.lib.recfunctions as recfuncs
import collections
import healpy

import ugali.utils.binning
import ugali.utils.skymap
from ugali.utils.projector import cel2gal, gal2cel
from ugali.utils.healpix import ang2pix, pix2ang, superpixel
from ugali.utils.shell import mkdir
from ugali.utils.logger import logger
from ugali.utils.config import Config

def pixelizeCatalog(infiles, config, force=False):
    """
    Break catalog up into a set of healpix files.
    """
    nside_catalog = config['coords']['nside_catalog']
    nside_pixel = config['coords']['nside_pixel']
    outdir = mkdir(config['catalog']['dirname'])
    filenames = config.getFilenames()
    
    for ii,infile in enumerate(infiles):
        logger.info('(%i/%i) %s'%(ii+1, len(infiles), infile))
        f = pyfits.open(infile)
        data = f[1].data
        header = f[1].header
        logger.info("%i objects found"%len(data))
        if not len(data): continue
        glon,glat = cel2gal(data['RA'],data['DEC'])
        catalog_pix = ang2pix(nside_catalog,glon,glat,coord='GAL')
        pixel_pix = ang2pix(nside_pixel,glon,glat,coord='GAL')
        names = [n.upper() for n in data.columns.names]
        ra_idx = names.index('RA'); dec_idx = names.index('DEC')
        idx = ra_idx if ra_idx > dec_idx else dec_idx
        catalog_pix_name = 'PIX%i'%nside_catalog
        pixel_pix_name = 'PIX%i'%nside_pixel

        coldefs = pyfits.ColDefs(
            [pyfits.Column(name='GLON',format='1D',array=glon),
             pyfits.Column(name='GLAT',format='1D',array=glat),
             pyfits.Column(name=catalog_pix_name,format='1J',array=catalog_pix),
             pyfits.Column(name=pixel_pix_name  ,format='1J',array=pixel_pix)]
        )
        hdu = pyfits.new_table(data.columns[:idx+1]+coldefs+data.columns[idx+1:])
        table = hdu.data

        for pix in numpy.unique(catalog_pix):
            logger.debug("Processing pixel %s"%pix)
            outfile = filenames.data['catalog'][pix]
            if not os.path.exists(outfile):
                logger.debug("Creating %s"%outfile)
                names = [n.upper() for n in table.columns.names]
                formats = table.columns.formats
                columns = [pyfits.Column(n,f) for n,f in zip(names,formats)]
                out = pyfits.HDUList([pyfits.PrimaryHDU(),pyfits.new_table(columns)])
                out[1].header['NSIDE'] = nside_catalog
                out[1].header['PIX'] = pix
                out.writeto(outfile)
            hdulist = pyfits.open(outfile,mode='update')
            t1 = hdulist[1].data
            # Could we speed up with sorting and indexing?
            t2 = table[ table[catalog_pix_name] == pix ]
            nrows1 = t1.shape[0]
            nrows2 = t2.shape[0]
            nrows = nrows1 + nrows2
            out = pyfits.new_table(t1.columns, nrows=nrows)
            for name in t1.columns.names:
                out.data.field(name)[nrows1:]=t2.field(name)
            hdulist[1] = out
            logger.debug("Writing %s"%outfile)
            hdulist.flush()
            hdulist.close()

def pixelizeDensity(config, nside=None, force=False):
    if nside is None: 
        nside = config['coords']['nside_likelihood']
    filenames = config.getFilenames()
    infiles = filenames[~filenames['catalog'].mask]

    for ii,f in enumerate(infiles.data):
        infile = f['catalog']
        pix = f['pix']
        logger.info('(%i/%i) %s'%(ii+1, len(infiles), infile))

        outfile = config['data']['density']%pix
        if os.path.exists(outfile) and not force: 
            logger.info("Found %s; skipping..."%outfile)
            continue
            
        outdir = mkdir(os.path.dirname(outfile))
        pixels, density = stellarDensity(infile,nside)
        data_dict = dict( DENSITY=density )
        logger.info("Writing %s..."%outfile)
        ugali.utils.skymap.writeSparseHealpixMap(pixels,data_dict,nside,outfile)

def stellarDensity(infile, nside=2**8): 
    area = healpy.nside2pixarea(nside,degrees=True)
    f = pyfits.open(infile)
    data = f[1].data
    logger.debug("Reading %s"%infile)
    
    glon,glat = data['GLON'],data['GLAT']
    pix = ang2pix(nside,glon,glat,coord='GAL')
    counts = collections.Counter(pix)
    pixels, number = numpy.array(sorted(counts.items())).T
    density = number/area
    f.close()

    return pixels, density

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()
