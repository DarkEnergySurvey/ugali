"""
Tools for making maps of the sky with healpix.
"""

import sys
import re
import gc

import numpy
import healpy
import pyfits

import ugali.utils.projector
from ugali.utils.healpix import superpixel,subpixel,ang2pix,pix2ang,query_disc
from ugali.utils.logger import logger
from ugali.utils.config import Config

############################################################

def surveyPixel(lon, lat, nside_pix, nside_subpix = None):
    """
    Return the set of HEALPix pixels that cover the given coordinates at resolution nside.
    Optionally return the set of subpixels within those pixels at resolution nside_subpix
    """
    pix = numpy.unique(ang2pix)
    if nside_subpix is None:
        return pix
    else:
        subpix_array = []
        for ii in range(0, len(pix)):
            subpix = subpixel(pix[ii], nside_pix, nside_subpix)
            subpix_array.append(subpix)
        return pix, numpy.array(subpix_array)

def inFootprint(config, pixels, nside=None):
    """
    Open each valid filename for the set of pixels and determine the set 
    of subpixels with valid data.
    """
    config = Config(config)
    nside_catalog    = config['coords']['nside_catalog']
    nside_likelihood = config['coords']['nside_likelihood']
    nside_pixel      = config['coords']['nside_pixel']

    if numpy.isscalar(pixels): pixels = numpy.array([pixels])
    if nside is None: nside = nside_likelihood

    filenames = config.getFilenames()
    catalog_pixels = filenames['pix'].compressed()

    inside = numpy.zeros(len(pixels), dtype=bool)
    if not nside_catalog:
        catalog_pix = [0]
    else:
        catalog_pix = superpixel(pixels,nside,nside_catalog)
        catalog_pix = numpy.intersect1d(catalog_pix,catalog_pixels)

    for fnames in filenames[catalog_pix]:
        logger.debug("Loading %s"%filenames['mask_1'])
        subpix_1,val_1 = ugali.utils.skymap.readSparseHealpixMap(fnames['mask_1'],'MAGLIM',construct_map=False)
        logger.debug("Loading %s"%fnames['mask_2'])
        subpix_2,val_2 = ugali.utils.skymap.readSparseHealpixMap(fnames['mask_2'],'MAGLIM',construct_map=False)
        subpix = numpy.intersect1d(subpix_1,subpix_2)
        superpix = numpy.unique(superpixel(subpix,nside_pixel,nside))
        inside |= numpy.in1d(pixels, superpix)
        
    return inside

def footprint(config, nside=None):
    """
    UNTESTED.
    Should return a boolean array representing the pixels in the footprint.
    """
    config = Config(config)
    if nside is None:
        nside = config['coords']['nside_pixel']
    elif nside < config['coords']['nside_catalog']:
        raise Exception('Requested nside=%i is greater than catalog_nside'%nside)
    elif nside > config['coords']['nside_pixel']:
        raise Exception('Requested nside=%i is less than pixel_nside'%nside)
    pix = numpy.arange(healpy.nside2npix(nside), dtype=int)
    return inFootprint(config,pix)


############################################################

def allSkyCoordinates(nside):
    """
    Generate a set of coordinates at the centers of pixels of resolutions nside across the full sky. 
    """
    lon,lat = pix2ang(nside, np.arange(0, healpy.nside2npix(nside)))
    return lon, lat

############################################################

def writeSparseHealpixMap(pix, data_dict, nside, outfile,
                          distance_modulus_array = None,
                          coordsys = 'NULL', ordering = 'NULL',
                          header_dict = None):
    """
    Sparse HEALPix maps are used to efficiently store maps of the sky by only
    writing out the pixels that contain data.

    Three-dimensional data can be saved by supplying a distance modulus array
    which is stored in a separate extension.
    
    coordsys [gal, cel]
    ordering [ring, nest]
    """

    hdul = pyfits.HDUList()

    # Pixel data extension
    columns_array = [pyfits.Column(name = 'PIX',
                                   format = 'K',
                                   array = pix)]

    for key in data_dict.keys():
        if data_dict[key].shape[0] != len(pix):
            logger.warning('First dimension of column %s (%i) does not match number of pixels (%i).'%(key,
                                                                                                      data_dict[key].shape[0],
                                                                                                      len(pix)))
        
        if len(data_dict[key].shape) == 1:
            columns_array.append(pyfits.Column(name = key,
                                               format = 'E',
                                               array = data_dict[key]))
        elif len(data_dict[key].shape) == 2:
            columns_array.append(pyfits.Column(name = key,
                                               format = '%iE'%(data_dict[key].shape[1]),
                                               array = data_dict[key]))
        else:
            logger.warning('Unexpected number of data dimensions for column %s.'%(key))
    
    hdu_pix_data = pyfits.new_table(columns_array)
    hdu_pix_data.header.update([('NSIDE', nside)])
    hdu_pix_data.header.update([('COORDSYS', coordsys.upper())])
    hdu_pix_data.header.update([('ORDERING', ordering.upper())])
    hdu_pix_data.header.update(header_dict)
    hdu_pix_data.name = 'PIX_DATA'
    hdul.append(hdu_pix_data)

    # Distance modulus extension
    if distance_modulus_array is not None:
        hdu_distance_modulus = pyfits.new_table([pyfits.Column(name = 'DISTANCE_MODULUS',
                                                               format = 'E',
                                                               array = distance_modulus_array)])
        hdu_distance_modulus.name = 'DISTANCE_MODULUS'
        hdul.append(hdu_distance_modulus)

    hdul.writeto(outfile, clobber = True)
    
############################################################

def readSparseHealpixMap(infile, field, extension='PIX_DATA', default_value=healpy.UNSEEN, construct_map=True):
    """
    Open a sparse HEALPix map fits file.
    Convert the contents into a HEALPix map or simply return the contents.
    Flexibility to handle 
    """
    reader = pyfits.open(infile,memmap=False)
    nside = reader[extension].header['NSIDE']

    # Trying to fix avoid a memory leak
    try:
        pix = numpy.array(reader[extension].data.field('PIX'),copy=True)
    except:
        pix = numpy.array(reader[extension].data.field('PIXEL'),copy=True)
    value = numpy.array(reader[extension].data.field(field),copy=True)
    reader.close()
    
    if construct_map:
        if len(value.shape) == 1:
            map = default_value * numpy.ones(healpy.nside2npix(nside))
            map[pix] = value
        else:
            map = default_value * numpy.ones([value.shape[1], healpy.nside2npix(nside)])
            for ii in range(0, value.shape[1]):
                map[ii][pix] = numpy.take(value, [ii], axis=1)
        ret = map
    else:
        if len(value.shape) == 1:
            ret = (pix,value)
        else:
            ret = (pix,value.transpose())
    return ret

############################################################

def readSparseHealpixMaps(infiles, field, extension='PIX_DATA', default_value=healpy.UNSEEN, construct_map=True):
    """
    Read multiple sparse healpix maps and output the results
    identically to a single file read.
    """
    if isinstance(infiles,basestring): infiles = [infiles]

    pix_array   = []
    value_array = []

    # Create a map based on the first file in the list
    map = readSparseHealpixMap(infiles[0], field, extension=extension, default_value=healpy.UNSEEN, construct_map=True)

    for ii in range(0, len(infiles)):
        logger.debug('(%i/%i) %s'%(ii+1, len(infiles), infiles[ii]))
        pix_array_current, value_array_current = readSparseHealpixMap(infiles[ii], field,
                                                                      extension=extension,
                                                                      construct_map=False)
        pix_array.append(pix_array_current)
        value_array.append(value_array_current)
        map[pix_array[ii]] = value_array[ii]

    # Check to see whether there are any conflicts
    pix_master = numpy.concatenate(pix_array)
    value_master = numpy.concatenate(value_array)

    n_conflicting_pixels = len(pix_master) - len(numpy.unique(pix_master)) 
    if n_conflicting_pixels != 0:
        logger.warning('%i conflicting pixels during merge.'%(n_conflicting_pixels))

    if construct_map:
        return map
    else:
        if n_conflicting_pixels == 0:
            pix_master = numpy.sort(pix_master)
            return pix_master, map[pix_master]
        else:
            pix_valid = numpy.nonzero(map != default_value)[0]
            return pix_valid, map[pix_valid]

############################################################

def mergeSparseHealpixMaps(infiles, outfile=None,
                           pix_data_extension='PIX_DATA',
                           pix_field='PIX',
                           distance_modulus_extension='DISTANCE_MODULUS',
                           distance_modulus_field='DISTANCE_MODULUS',
                           default_value=healpy.UNSEEN):
    """
    Use the first infile to determine the basic contents to expect for the other files.
    """
    # Setup
    if isinstance(infiles,basestring): infiles = [infiles]
    
    distance_modulus_array = None
    pix_array = []
    data_dict = {}

    reader = pyfits.open(infiles[0])
    nside = reader[pix_data_extension].header['NSIDE']

    for ii in range(0, len(reader)):
        if reader[ii].name == distance_modulus_extension:
            distance_modulus_array = reader[distance_modulus_extension].data.field(distance_modulus_field)

    for key in reader[pix_data_extension].data.names:
        if key == pix_field:
            continue
        data_dict[key] = []
        #if distance_modulus_array is None:
        #    data_dict[key] = default_value * numpy.ones(healpy.nside2npix(nside))
        #else:
        #    data_dict[key] = default_value * numpy.ones([len(distance_modulus_array),
        #                                                 healpy.nside2npix(nside)])
    reader.close()

    # Now loop over the infiles

    for ii in range(0, len(infiles)):
        logger.debug('(%i/%i) %s'%(ii+1, len(infiles), infiles[ii]))

        reader = pyfits.open(infiles[ii])
        distance_modulus_array_current = numpy.array(reader[distance_modulus_extension].data.field(distance_modulus_field),copy=True)
        if not numpy.array_equal(distance_modulus_array_current,distance_modulus_array):
            logger.warning("Distance moduli do not match; skipping...")
            continue
        reader.close()

        pix_array_current = readSparseHealpixMap(infiles[ii], pix_field,
                                                 extension=pix_data_extension, construct_map=False)[0]
        pix_array.append(pix_array_current)

        for key in data_dict.keys():
            value_array_current = readSparseHealpixMap(infiles[ii], key,
                                                       extension=pix_data_extension, construct_map=False)[1]
            data_dict[key].append(value_array_current)
            #if distance_modulus_array is None:
            #    data_dict[key][pix_array_current] = value
            #else:
            #    for jj in range(0, len(distance_modulus_array)):
            #        data_dict[key][jj] = value[jj]

        gc.collect()

    pix_master = numpy.concatenate(pix_array)
    n_conflicting_pixels = len(pix_master) - len(numpy.unique(pix_master)) 
    if n_conflicting_pixels != 0:
        logger.warning('%i conflicting pixels during merge.'%(n_conflicting_pixels))

    for key in data_dict.keys():
        if distance_modulus_array is not None:
            data_dict[key] = numpy.concatenate(data_dict[key], axis=1).transpose()
        else:
            data_dict[key] = numpy.concatenate(data_dict[key])

    if outfile is not None:
        writeSparseHealpixMap(pix_master, data_dict, nside, outfile,
                              distance_modulus_array=distance_modulus_array,
                              coordsys='NULL', ordering='NULL')
    else:
        return data_dict


############################################################

def mergeLikelihoodFiles(infiles, lkhdfile, roifile):
    mergeSparseHealpixMaps(infiles,lkhdfile)

    ext='PIX_DATA'
    keys=['STELLAR','NINSIDE','NANNULUS']
    nside = pyfits.open(infiles[0])[ext].header['LKDNSIDE']

    pix_array = []
    data_dict = dict([(k,[]) for k in keys])
    for ii in range(0, len(infiles)):
        logger.debug('(%i/%i) %s'%(ii+1, len(infiles), infiles[ii]))
        reader = pyfits.open(infiles[ii])
        pix_array.append(reader[ext].header['LKDPIX'])
        for key in data_dict.keys():
            data_dict[key].append(reader[ext].header[key])
        
    pix_array = numpy.array(pix_array)
    for key in data_dict.keys():
        data_dict[key] = numpy.array(data_dict[key])
    writeSparseHealpixMap(pix_array, data_dict, nside, roifile)

############################################################

def randomPositions(input, nside_pix, n=1):
    """
    Generate n random positions within a full HEALPix mask of booleans, or a set of (lon, lat) coordinates.

    nside_pix is meant to be at coarser resolution than the input mask or catalog object positions
    so that gaps from star holes, bleed trails, cosmic rays, etc. are filled in. 
    Return the longitude and latitude of the random positions and the total area (deg^2).

    Probably there is a faster algorithm, but limited much more by the simulation and fitting time
    than by the time it takes to generate random positions within the mask.
    """
    input = numpy.array(input)
    if len(input.shape) == 1:
        subpix = numpy.nonzero(input)[0] # All the valid pixels in the mask at the NSIDE for the input mask
        lon, lat = pix2ang(healpy.npix2nside(len(input)), subpix)
    elif len(input.shape) == 2:
        lon, lat = input[0], input[1] # All catalog object positions
    else:
        logger.warning('Unexpected input dimensions for skymap.randomPositions')
    pix = surveyPixel(lon, lat, nside_pix)

    # Area with which the random points are thrown
    area = len(pix) * healpy.nside2pixarea(nside_pix, degrees=True)
    
    lon = []
    lat = []
    for ii in range(0, n):
        # Choose an unmasked pixel at random, which is OK because HEALPix is an equal area scheme
        pix_ii = pix[numpy.random.randint(0, len(pix))]
        lon_ii, lat_ii = ugali.utils.projector.pixToAng(nside_pix, pix_ii)
        projector = ugali.utils.projector.Projector(lon_ii, lat_ii)

        inside = False
        while not inside:
            # Apply random offset
            arcminToDegree = 1 / 60.
            resolution = arcminToDegree * healpy.nside2resol(nside_pix, arcmin=True)
            x = 2. * (numpy.random.rand() - 0.5) * resolution # Using factor 2 to be conservative
            y = 2. * (numpy.random.rand() - 0.5) * resolution
            
            lon_candidate, lat_candidate = projector.imageToSphere(x, y)

            # Make sure that the random position does indeed fall within the randomly selected pixel 
            if ugali.utils.projector.angToPix(nside_pix, lon_candidate, lat_candidate) == pix_ii:
                inside = True
                                    
        lon.append(lon_candidate)
        lat.append(lat_candidate)

    return numpy.array(lon), numpy.array(lat), area

############################################################

