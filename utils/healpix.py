#!/usr/bin/env python
"""
toolkit for working with healpix
"""
import numpy
import healpy


############################################################

def superpixel(subpix, nside_subpix, nside_superpix):
    """
    Return the indices of the super-pixels which contain each of the sub-pixels.
    """
    theta, phi =  healpy.pix2ang(nside_subpix, subpix)
    return healpy.ang2pix(nside_superpix, theta, phi)

def subpixel(superpix, nside_superpix, nside_subpix):
    """
    Return the indices of sub-pixels (resolution nside_subpix) within the super-pixel with (resolution nside_superpix).
    """
    vec = healpy.pix2vec(nside_superpix, superpix)
    radius = numpy.degrees(2. * healpy.max_pixrad(nside_superpix))
    subpix = query_disc(nside_subpix, vec, radius)
    pix_for_subpix = superpixel(subpix,nside_subpix,nside_superpix)
    # Might be able to speed up array indexing...
    return subpix[pix_for_subpix == superpix]

############################################################

def pixToAng(nside, pix):
    """
    Return (lon, lat) in degrees instead of (theta, phi) in radians
    """
    theta, phi =  healpy.pix2ang(nside, pix)
    lon = numpy.degrees(phi)
    lat = 90. - numpy.degrees(theta)                    
    return lon, lat

pix2ang = pixToAng

def angToPix(nside, lon, lat, coord='GAL'):
    """
    Input (lon, lat) in degrees instead of (theta, phi) in radians
    """
    theta = numpy.radians(90. - lat)
    phi = numpy.radians(lon)
    return healpy.ang2pix(nside, theta, phi)

ang2pix = angToPix

def angToVec(lon, lat):
    vec = healpy.ang2vec(numpy.radians(90. - lat), numpy.radians(lon))
    return vec

ang2vec = angToVec

############################################################

def in_pixels(lon,lat,pixels,nside):
    """
    Check if (lon,lat) in pixel list.
    """
    pix = ang2pix(nside,lon,lat)
    return numpy.in1d(pix,pixels)

def index_pixels(lon,lat,pixels,nside):
   """
   Find the index for object amoung a subset of healpix pixels.
   Set index of objects outside the pixel subset to -1

   # ADW: Not really safe to set index = -1 (accesses last entry); 
   # -np.inf would be better, but breaks code...
   """
   pix = ang2pix(nside,lon,lat)

   # pixels should be pre-sorted, otherwise...
   index = numpy.searchsorted(pixels,pix)
   # Find objects that are outside the roi
   #index[numpy.take(pixels,index,mode='clip')!=pix] = -1
   index[~numpy.in1d(pix,pixels)] = -1
   return index

############################################################

def query_disc(nside, vec, radius, inclusive=False, fact=4, nest=False):
    """
    Wrapper around healpy.query_disc to deal with old healpy implementation.

    nside : int
      The nside of the Healpix map.
    vec : float, sequence of 3 elements
      The coordinates of unit vector defining the disk center.
    radius : float
      The radius (in degrees) of the disk
    inclusive : bool, optional
      If False, return the exact set of pixels whose pixel centers lie 
      within the disk; if True, return all pixels that overlap with the disk,
      and maybe a few more. Default: False
    fact : int, optional
      Only used when inclusive=True. The overlapping test will be done at
      the resolution fact*nside. For NESTED ordering, fact must be a power of 2,
      else it can be any positive integer. Default: 4.
    nest: bool, optional
      if True, assume NESTED pixel ordering, otherwise, RING pixel ordering

    """
    try: 
        # New-style call (healpy 1.6.3)
        return healpy.query_disc(nside, vec, numpy.radians(radius), inclusive, fact, nest)
    except: 
        # Old-style call (healpy 0.10.2)
        return healpy.query_disc(nside, vec, numpy.radians(radius), nest, deg=False)

############################################################


if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args
