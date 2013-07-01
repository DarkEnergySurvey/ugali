"""
Tools for making maps of the sky with healpix.
"""

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

def writeSparseHealpixMap(pix, value, nside, outfile,
                          coordsys = 'NULL', ordering = 'NULL'):
    """
    coordsys [gal, cel]
    ordering [ring, nest]
    """

    columns_array = [pyfits.Column(name = 'PIX',
                                   format = 'K',
                                   array = pix),
                     pyfits.Column(name = 'VALUE',
                                   format = 'E',
                                   array = value)]
    
    hdu = pyfits.new_table(columns_array)
    hdu.header.update('NSIDE', nside)
    hdu.header.update('COORDSYS', coordsys)
    hdu.header.update('ORDERING', ordering)
    hdu.writeto(outfile, clobber = True)
    
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
