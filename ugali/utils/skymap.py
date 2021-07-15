"""
Tools for making maps of the sky with healpix. Used by simulations.
"""

import numpy as np
import healpy as hp

import ugali.utils.projector
from ugali.utils.healpix import superpixel,subpixel,ang2pix,pix2ang,query_disc
from ugali.utils.healpix import read_partial_map
from ugali.utils.logger import logger
from ugali.utils.config import Config

############################################################

def surveyPixel(lon, lat, nside_pix, nside_subpix = None):
    """
    Return the set of HEALPix pixels that cover the given coordinates at resolution nside.
    Optionally return the set of subpixels within those pixels at resolution nside_subpix
    """
    pix = np.unique(ang2pix(nside_pix, lon, lat))
    if nside_subpix is None:
        return pix
    else:
        subpix_array = []
        for ii in range(0, len(pix)):
            subpix = subpixel(pix[ii], nside_pix, nside_subpix)
            subpix_array.append(subpix)
        return pix, np.array(subpix_array)

def inFootprint(config, pixels, nside=None):
    """
    Open each valid filename for the set of pixels and determine the set 
    of subpixels with valid data.

    Parameters
    ----------
    config : config
        Configuration (file or object)
    pixels : array or int
        List of pixels to create footprint for
    nside  : int, optional
        Healpix nside
        
    Returns
    -------
    inside : array
        Boolean array of whether pixel is in footprint
    """
    logger.info("Calculating survey footprint...")

    config = Config(config)
    nside_catalog    = config['coords']['nside_catalog']
    nside_likelihood = config['coords']['nside_likelihood']
    nside_pixel      = config['coords']['nside_pixel']

    if np.isscalar(pixels): pixels = np.array([pixels])
    if nside is None: nside = nside_likelihood

    filenames = config.getFilenames()
    catalog_pixels = filenames['pix'].compressed()

    inside = np.zeros(len(pixels), dtype=bool)
    if not nside_catalog:
        catalog_pix = [0]
    else:
        catalog_pix = superpixel(pixels,nside,nside_catalog)
        catalog_pix = np.intersect1d(catalog_pix,catalog_pixels)

    fnames = filenames[catalog_pix]

    # Load the first mask
    logger.debug("Loading %s"%fnames['mask_1'])
    _nside,subpix1,val1 = read_partial_map(fnames['mask_1'],'MAGLIM',
                                           fullsky=False,multiproc=8)
    # Load the second mask
    logger.debug("Loading %s"%fnames['mask_2'])
    _nside,subpix2,val2 = read_partial_map(fnames['mask_2'],'MAGLIM',
                                           fullsky=False,multiproc=8)
    # Run the subpixels
    subpix = np.intersect1d(subpix1,subpix2)
    superpix = np.unique(superpixel(subpix,nside_pixel,nside))
    inside |= np.in1d(pixels, superpix)
    
    return inside

############################################################

def allSkyCoordinates(nside):
    """
    Generate a set of coordinates at the centers of pixels of resolutions nside across the full sky. 
    """
    lon,lat = pix2ang(nside, np.arange(0, hp.nside2npix(nside)))
    return lon, lat

def coarseFootprint(input, nside_pix):
    """
    Generate a coarse healpix mask of booleans from a finer healpix
    mask or a set of (lon, lat) coordinates.

    Parameters:
    -----------
    input :     (1) full HEALPix mask of booleans, or (2) a set of (lon, lat) coordinates for catalog objects that define the occupied pixels.
    nside_pix : nside_pix is meant to be at coarser (or equivalent) resolution than the input mask or catalog object positions
    so that gaps from star holes, bleed trails, cosmic rays, etc. are filled in. 

    Returns:
    --------
    lon,lat,area : Return the longitude and latitude of the random positions (deg) and the total area (deg^2).
    """
    input = np.array(input)
    if len(input.shape) == 1:
        if hp.npix2nside(len(input)) < nside_pix:
            logger.warning('Expected coarser resolution nside_pix in skymap.randomPositions')
        subpix = np.nonzero(input)[0] # All the valid pixels in the mask at the NSIDE for the input mask
        lon, lat = pix2ang(hp.npix2nside(len(input)), subpix)
    elif len(input.shape) == 2:
        lon, lat = input[0], input[1] # All catalog object positions
    else:
        logger.warning('Unexpected input dimensions for skymap.randomPositions')
    pix = surveyPixel(lon, lat, nside_pix)

    # Area with which the random points are thrown
    area = len(pix) * hp.nside2pixarea(nside_pix, degrees=True)

    # Create mask at the coarser resolution
    mask = np.tile(False, hp.nside2npix(nside_pix))
    mask[pix] = True
    
    return mask

def randomPositions(input, nside_pix, n=1):
    """
    Generate n random positions within a full HEALPix mask of booleans, or a set of (lon, lat) coordinates.

    Parameters:
    -----------
    input :     (1) full HEALPix mask of booleans, or (2) a set of (lon, lat) coordinates for catalog objects that define the occupied pixels.
    nside_pix : nside_pix is meant to be at coarser resolution than the input mask or catalog object positions
    so that gaps from star holes, bleed trails, cosmic rays, etc. are filled in. 
    n     : number of random points

    Returns:
    --------
    lon,lat,area : Return the longitude and latitude of the random positions (deg) and the total area (deg^2).

    """
    mask = coarseFootprint(input, nside_pix)
    area = mask.sum() * hp.nside2pixarea(nside_pix, degrees=True)

    # Estimate the number of points that need to be thrown based off
    # coverage fraction of the HEALPix mask
    coverage_fraction = float(np.sum(mask)) / len(mask) 
    n_throw = int(n / coverage_fraction)
        
    lon, lat = [], []
    count = 0
    while len(lon) < n:
        lon_throw = np.random.uniform(0., 360., n_throw)
        lat_throw = np.degrees(np.arcsin(np.random.uniform(-1., 1., n_throw)))

        pix_throw = ugali.utils.healpix.angToPix(nside_pix, lon_throw, lat_throw)
        cut = mask[pix_throw].astype(bool)

        lon = np.append(lon, lon_throw[cut])
        lat = np.append(lat, lat_throw[cut])

        count += 1
        if count > 10:
            raise RuntimeError('Too many loops...')

    return lon[0:n], lat[0:n], area

############################################################

def randomPositionsMask(mask, nside_pix, n):
    """
    Generate n random positions within a HEALPix mask of booleans.

    KCB: Likely superceded by the randomPositions function, but more generic.
    """
    
    npix = len(mask)
    nside = hp.npix2nside(npix)

    # Estimate the number of points that need to be thrown based off
    # coverage fraction of the HEALPix mask
    coverage_fraction = float(np.sum(mask)) / len(mask) 
    n_throw = int(n / coverage_fraction)
        
    lon, lat = [], []
    latch = True
    count = 0
    while len(lon) < n:
        lon_throw = np.random.uniform(0., 360., n_throw)
        lat_throw = np.degrees(np.arcsin(np.random.uniform(-1., 1., n_throw)))

        pix = ugali.utils.healpix.angToPix(nside, lon_throw, lat_throw)
        cut = mask[pix].astype(bool)

        lon = np.append(lon, lon_throw[cut])
        lat = np.append(lat, lat_throw[cut])

        count += 1
        if count > 10:
            raise RuntimeError('Too many loops...')

    return lon[0:n], lat[0:n]

############################################################
