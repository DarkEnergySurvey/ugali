"""
Tools for making maps of the sky with healpix.
"""

import sys
import numpy
import pyfits
import healpy

############################################################

def subpixel(pix, nside_pix, nside_subpix):
    """
    Return the pixel indices at resolution nside_subpix that are within pixel pix at resolution nside_pix.
    """
    vec = healpy.pix2vec(nside_pix, pix)
    radius = numpy.degrees(2. * healpy.nside2resol(nside_pix))
    subpix = healpy.query_disc(nside_subpix, vec, radius)
    theta, phi =  healpy.pix2ang(nside_subpix, subpix)
    pix_for_subpix = healpy.ang2pix(nside_pix, theta, phi)
    return subpix[pix_for_subpix == pix]
    
############################################################

def surveyPixel(lon, lat, nside_pix, nside_subpix = None):
    """
    Return the set of HEALPix pixels that cover the given coordinates at resolution nside.
    Optionally return the set of subpixels within those pixels at resolution nside_subpix
    """
    theta = numpy.radians(90. - lat)
    phi = numpy.radians(lon)
    pix = numpy.unique(healpy.ang2pix(nside_pix, theta, phi))
    if nside_subpix is None:
        return pix
    else:
        subpix_array = []
        for ii in range(0, len(pix)):
            subpix = subpixel(pix[ii], nside_pix, nside_subpix)
            subpix_array.append(subpix)
        return pix, numpy.array(subpix_array)

############################################################

def allSkyCoordinates(nside):
    """
    Generate a set of coordinates at the centers of pixels of resolutions nside across the full sky. 
    """
    theta, phi =  healpy.pix2ang(nside, range(0, healpy.nside2npix(nside)))
    lon = numpy.degrees(phi)
    lat = 90. - numpy.degrees(theta)                    
    return lon, lat

############################################################

def writeSparseHealpixMap(pix, data_dict, nside, outfile,
                          distance_modulus_array = None,
                          coordsys = 'NULL', ordering = 'NULL'):
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
            print 'WARNING: first dimension of column %s (%i) does not match number of pixels (%i).'%(key,
                                                                                                      data_dict[key].shape[0],
                                                                                                      len(pix))
        
        if len(data_dict[key].shape) == 1:
            columns_array.append(pyfits.Column(name = key,
                                               format = 'E',
                                               array = data_dict[key]))
        elif len(data_dict[key].shape) == 2:
            columns_array.append(pyfits.Column(name = key,
                                               format = '%iE'%(data_dict[key].shape[1]),
                                               array = data_dict[key]))
        else:
            print 'WARNING: unexpected number of data dimensions for column %s.'%(key)
    
    hdu_pix_data = pyfits.new_table(columns_array)
    hdu_pix_data.header.update('NSIDE', nside)
    hdu_pix_data.header.update('COORDSYS', coordsys.upper())
    hdu_pix_data.header.update('ORDERING', ordering.upper())
    hdu_pix_data.name = 'PIX_DATA'
    hdul.append(hdu_pix_data)

    # Distance modulus extension
    if distance_modulus_array is not None:
        hdu_distance_modulus = pyfits.new_table(pyfits.Column(name = 'DISTANCE_MODULUS',
                                                              format = 'E',
                                                              array = distance_modulus_array))
        hdu_distance_modulus.name = 'DISTANCE_MODULUS'
        hdul = pyfits.append(hdu_distance_modulus)

    hdul.writeto(outfile, clobber = True)
    
############################################################

def readSparseHealpixMap(infile, extension=1, default_value=healpy.UNSEEN, construct_map=True):
    """
    Open a sparse HEALPix map fits file.
    Convert the contents into a HEALPix map or simply return the contents.
    """
    reader = pyfits.open(infile)
    pix = reader[extension].data.field('PIX')
    value = reader[extension].data.field('VALUE')

    if construct_map:
        map = default_value * numpy.ones(healpy.nside2npix(reader[extension].header['NSIDE']))
        map[pix] = value
        reader.close()
        return map
    else:
        reader.close()
        return pix, value

############################################################

def mergeSparseHealpixMaps(infiles, extension=1, default_value=healpy.UNSEEN, construct_map=True):
    """
    Ideas: store only the pixels with data for each roi, merge results later
    """

    pix_array = [] #* len(infiles)
    value_array = [] #* len(infiles)

    # Create a map based on the first file in the list
    map = readSparseHealpixMap(infiles[0], extension=extension, default_value=healpy.UNSEEN, construct_map=True)

    for ii in range(0, len(infiles)):
        print '(%i/%i) %s'%(ii, len(infiles), infiles[ii])
        pix_array_current, value_array_current = readSparseHealpixMap(infiles[ii],
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
        print 'WARNING: %i conflicting pixels during merge.'%(n_conflicting_pixels)

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
