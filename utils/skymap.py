"""
Tools for making maps of the sky with healpix.
"""

import sys
import numpy
import pyfits
import healpy

import ugali.utils.projector

############################################################

def subpixel(pix, nside_pix, nside_subpix):
    """
    Return the indices of pixels with resolution nside_subpix that are within larger pixel pix with resolution nside_pix.
    """
    vec = healpy.pix2vec(nside_pix, pix)
    radius = 2. * healpy.nside2resol(nside_pix)
    subpix = ugali.utils.projector.query_disc(nside_subpix, vec, radius)
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
    reader = pyfits.open(infile)
    pix = reader[extension].data.field('PIX')
    value = reader[extension].data.field(field)
    
    if construct_map:
        if len(value.shape) == 1:
            map = default_value * numpy.ones(healpy.nside2npix(reader[extension].header['NSIDE']))
            map[pix] = value
        else:
            map = default_value * numpy.ones([value.shape[1], healpy.nside2npix(reader[extension].header['NSIDE'])])
            for ii in range(0, value.shape[1]):
                map[ii][pix] = numpy.take(value, [ii], axis=1)
        reader.close()
        return map
    else:
        reader.close()
        if len(value.shape) == 1:
            return pix, value
        else:
            return pix, value.transpose()

############################################################

def mergeSparseHealpixMaps2(infiles, extension='PIX_DATA', default_value=healpy.UNSEEN, construct_map=True):
    """
    Ideas: store only the pixels with data for each roi, merge results later
    """

    # TODO: THIS FUNCTION NEEDS TO BE GENERALIZED TO HANDLE 3D MAPS AND MULTIPLE FIELDS

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

def mergeSparseHealpixMaps(infiles, outfile,
                           pix_data_extension='PIX_DATA',
                           pix_field='PIX',
                           distance_modulus_extension='DISTANCE_MODULUS',
                           distance_modulus_field='DISTANCE_MODULUS',
                           default_value=healpy.UNSEEN):
    """
    Use the first infile to determine the basic contents to expect for the other files.
    """

    # Setup
    
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
        print '(%i/%i) %s'%(ii, len(infiles), infiles[ii])

        pix_array_current = readSparseHealpixMap(infiles[ii], pix_field,
                                                 extension=pix_data_extension, construct_map=False)[0]
        pix_array.append(pix_array_current)

        for key in data_dict.keys():
            data_dict[key].append(readSparseHealpixMap(infiles[ii], key,
                                                       extension=pix_data_extension, construct_map=False)[1])
            #if distance_modulus_array is None:
            #    data_dict[key][pix_array_current] = value
            #else:
            #    for jj in range(0, len(distance_modulus_array)):
            #        data_dict[key][jj] = value[jj]

    pix_master = numpy.concatenate(pix_array)
    n_conflicting_pixels = len(pix_master) - len(numpy.unique(pix_master)) 
    if n_conflicting_pixels != 0:
        print 'WARNING: %i conflicting pixels during merge.'%(n_conflicting_pixels)

    for key in data_dict.keys():
        if distance_modulus_array is not None:
            data_dict[key] = numpy.concatenate(data_dict[key], axis=1).transpose()
        else:
            data_dict[key] = numpy.concatenate(data_dict[key])
    
    writeSparseHealpixMap(pix_master, data_dict, nside, outfile,
                          distance_modulus_array=distance_modulus_array,
                          coordsys='NULL', ordering='NULL')

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
        lon, lat = ugali.utils.projector.pixToAng(healpy.npix2nside(len(input)), subpix)
    elif len(input.shape) == 2:
        lon, lat = input[0], input[1] # All catalog object positions
    else:
        print 'WARNING: unexpected input dimensions for skymap.randomPositions'
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

