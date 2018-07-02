#!/usr/bin/env python
"""
For pixelizing catalogs and masks.
"""

#FIXME: ADW This needs to be updated to use fitsio

import os
from os.path import join
import glob
import collections

import fitsio
import numpy as np
import numpy.lib.recfunctions as recfuncs
import healpy as hp

#import ugali.utils.binning
#import ugali.utils.skymap
from ugali.utils.projector import cel2gal, gal2cel
from ugali.utils import healpix, mlab
from ugali.utils.healpix import ang2pix, pix2ang, superpixel

from ugali.utils.shell import mkdir
from ugali.utils.logger import logger
from ugali.utils.config import Config
import ugali.utils.fileio

def pixelizeCatalog(infiles, config, force=False):
    """
    Break catalog into chunks by healpix pixel.
    
    Parameters:
    -----------
    infiles : List of input files
    config  : Configuration file
    force   : Overwrite existing files (depricated)
    
    Returns:
    --------
    None
    """
    nside_catalog = config['coords']['nside_catalog']
    nside_pixel = config['coords']['nside_pixel']
    coordsys = config['coords']['coordsys'].upper()
    outdir = mkdir(config['catalog']['dirname'])
    filenames = config.getFilenames()
    lon_field = config['catalog']['lon_field'].upper()
    lat_field = config['catalog']['lat_field'].upper()

    # ADW: It would probably be better (and more efficient) to do the
    # pixelizing and the new column insertion separately.
    for i,filename in enumerate(infiles):
        logger.info('(%i/%i) %s'%(i+1, len(infiles), filename))
        data = fitsio.read(filename)
        logger.info("%i objects found"%len(data))
        if not len(data): continue

        columns = map(str.upper,data.dtype.names)
        names,arrs = [],[]

        if (lon_field in columns) and (lat_field in columns):
            lon,lat = data[lon_field],data[lat_field]
        elif coordsys == 'GAL':
            msg = "Columns '%s' and '%s' not found."%(lon_field,lat_field)
            msg += "\nConverting from RA,DEC"
            logger.warning(msg)
            lon,lat = cel2gal(data['RA'],data['DEC'])
            names += [lon_field,lat_field]
            arrs  += [lon,lat]
        elif coordsys == 'CEL':
            msg = "Columns '%s' and '%s' not found."%(lon_field,lat_field)
            msg += "\nConverting from GLON,GLAT"
            lon,lat = gal2cel(data['GLON'],data['GLAT'])
            names  += [lon_field,lat_field]
            arrs   += [lon,lat]

        cat_pix = ang2pix(nside_catalog,lon,lat)
        pix_pix = ang2pix(nside_pixel,lon,lat)
        cat_pix_name = 'PIX%i'%nside_catalog
        pix_pix_name = 'PIX%i'%nside_pixel

        try:
            names += [cat_pix_name,pix_pix_name]
            arrs  += [cat_pix,pix_pix]
            data=mlab.rec_append_fields(data,names=names,arrs=arrs)
        except ValueError as e:
            logger.warn(str(e)+'; not adding column.')
            #data[cat_pix_name] = cat_pix
            #data[pix_pix_name] = pix_pix
                               
        for pix in np.unique(cat_pix):
            logger.debug("Processing pixel %s"%pix)

            arr = data[cat_pix == pix]
            outfile = filenames.data['catalog'][pix]

            if not os.path.exists(outfile):
                logger.debug("Creating %s"%outfile)
                out=fitsio.FITS(outfile,mode='rw')
                out.write(arr)

                hdr=healpix.header_odict(nside=nside_catalog,
                                                     coord=coordsys[0])
                for key in ['PIXTYPE','ORDERING','NSIDE','COORDSYS']:
                    out[1].write_key(*list(hdr[key].values()))
                out[1].write_key('PIX',pix,comment='HEALPIX pixel for this file')
            else:
                out=fitsio.FITS(outfile,mode='rw')
                out[1].append(arr)

            logger.debug("Writing %s"%outfile)
            out.close()

def pixelizeDensity(config, nside=None, force=False):
    if nside is None: 
        nside = config['coords']['nside_likelihood']
    coordsys = config['coords']['coordsys'].upper()
    filenames = config.getFilenames()
    infiles = filenames[~filenames['catalog'].mask]
    lon_field = config['catalog']['lon_field'].upper()
    lat_field = config['catalog']['lat_field'].upper()

    for ii,f in enumerate(infiles.data):
        infile = f['catalog']
        pix = f['pix']
        logger.info('(%i/%i) %s'%(ii+1, len(infiles), infile))

        outfile = config['data']['density']%pix
        if os.path.exists(outfile) and not force: 
            logger.info("Found %s; skipping..."%outfile)
            continue
            
        outdir = mkdir(os.path.dirname(outfile))
        pixels, density = stellarDensity(infile,nside,
                                         lon_field=lon_field,lat_field=lat_field)

        data = dict(PIXEL=pixels,DENSITY=density)
        healpix.write_partial_map(outfile,data,nside=nside,coord=coordsys[0])


def stellarDensity(infile, nside=256, lon_field='RA', lat_field='DEC'): 
    area = hp.nside2pixarea(nside,degrees=True)
    logger.debug("Reading %s"%infile)
    data = fitsio.read(infile,columns=[lon_field,lat_field])

    lon,lat = data[lon_field],data[lat_field]
    pix = ang2pix(nside,lon,lat)
    counts = collections.Counter(pix)
    pixels, number = np.array(sorted(counts.items())).T
    density = number/area

    return pixels, density

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()
