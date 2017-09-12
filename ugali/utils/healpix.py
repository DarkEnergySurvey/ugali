#!/usr/bin/env python
"""
Toolkit for working with healpix
"""

from collections import OrderedDict as odict

import numpy
import numpy as np
import healpy as hp
import healpy
import fitsio

import ugali.utils.fileio
from ugali.utils.logger import logger

############################################################

def superpixel(subpix, nside_subpix, nside_superpix):
    """
    Return the indices of the super-pixels which contain each of the sub-pixels.
    """
    if nside_subpix==nside_superpix: return subpix
    theta, phi =  hp.pix2ang(nside_subpix, subpix)
    return hp.ang2pix(nside_superpix, theta, phi)

def subpixel(superpix, nside_superpix, nside_subpix):
    """
    Return the indices of sub-pixels (resolution nside_subpix) within the super-pixel with (resolution nside_superpix).
    """
    if nside_superpix==nside_subpix: return superpix
    vec = hp.pix2vec(nside_superpix, superpix)
    radius = np.degrees(2. * hp.max_pixrad(nside_superpix))
    subpix = query_disc(nside_subpix, vec, radius)
    pix_for_subpix = superpixel(subpix,nside_subpix,nside_superpix)
    # Might be able to speed up array indexing...
    return subpix[pix_for_subpix == superpix]

############################################################

def phi2lon(phi): return np.degrees(phi)
def lon2phi(lon): return np.radians(lon)

def theta2lat(theta): return 90. - np.degrees(theta)
def lat2theta(lat): return np.radians(90. - lat)

def pix2ang(nside, pix, nest=False):
    """
    Return (lon, lat) in degrees instead of (theta, phi) in radians
    """
    theta, phi =  hp.pix2ang(nside, pix, nest=nest)
    lon = phi2lon(phi)
    lat = theta2lat(theta)
    return lon, lat

def ang2pix(nside, lon, lat, nest=False):
    """
    Input (lon, lat) in degrees instead of (theta, phi) in radians
    """
    theta = np.radians(90. - lat)
    phi = np.radians(lon)
    return healpy.ang2pix(nside, theta, phi, nest=nest)

def ang2vec(lon, lat):
    theta = lat2theta(lat)
    phi = lon2phi(lon)
    vec = hp.ang2vec(theta, phi)
    return vec

pixToAng = pix2ang
angToPix = ang2pix
angToVec = ang2vec

############################################################

def healpixMap(nside, lon, lat, fill_value=0.):
    """
    Input (lon, lat) in degrees instead of (theta, phi) in radians.
    Returns HEALPix map at the desired resolution 
    """
    pix = angToPix(nside, lon, lat)
    m = np.histogram(pix, np.arange(hp.nside2npix(nside) + 1))[0].astype(float)
    if fill_value != 0.:
        m[m == 0.] = fill_value
    return m

############################################################

def in_pixels(lon,lat,pixels,nside):
    """
    Check if (lon,lat) in pixel list.
    """
    pix = ang2pix(nside,lon,lat)
    return np.in1d(pix,pixels)

def index_pixels(lon,lat,pixels,nside):
   """
   Find the index for object amoung a subset of healpix pixels.
   Set index of objects outside the pixel subset to -1

   # ADW: Not really safe to set index = -1 (accesses last entry); 
   # -np.inf would be better, but breaks other code...
   """
   pix = ang2pix(nside,lon,lat)
   # pixels should be pre-sorted, otherwise...???
   index = np.searchsorted(pixels,pix)
   if np.isscalar(index):
       if not np.in1d(pix,pixels).any(): index = -1
   else:
       # Find objects that are outside the roi
       #index[np.take(pixels,index,mode='clip')!=pix] = -1
       index[~np.in1d(pix,pixels)] = -1
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
      The radius (in degrees) of the disc
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
        return hp.query_disc(nside, vec, np.radians(radius), inclusive, fact, nest)
    except Exception as e: 
        print e
        # Old-style call (healpy 0.10.2)
        return hp.query_disc(nside, vec, np.radians(radius), nest, deg=False)

def ang2disc(nside, lon, lat, radius, inclusive=False, fact=4, nest=False):
    """
    Wrap `query_disc` to use lon, lat, and radius in degrees.
    """
    vec = ang2vec(lon,lat)
    return query_disc(nside,vec,radius,inclusive,fact,nest)

angToDisc = ang2disc

def get_interp_val(m, lon, lat, *args, **kwargs):
    return hp.get_interp_val(m, lat2theta(lat), lon2phi(lon), *args, **kwargs)

############################################################

def header_odict(nside,nest=False,ordering='RING',coord=None, partial=True):
    """Mimic the healpy header keywords."""
    hdr = odict([])
    hdr['PIXTYPE']=odict([('name','PIXTYPE'),
                          ('value','HEALPIX'),
                          ('comment','HEALPIX pixelisation')])

    ordering = 'NEST' if nest else 'RING'
    hdr['ORDERING']=odict([('name','ORDERING'),
                           ('value',ordering),
                           ('comment','Pixel ordering scheme, either RING or NESTED')])
    hdr['NSIDE']=odict([('name','NSIDE'),
                        ('value',nside),
                        ('comment','Resolution parameter of HEALPIX')])
    if coord:
        hdr['COORDSYS']=odict([('name','COORDSYS'), 
                               ('value',coord), 
                               ('comment','Ecliptic, Galactic or Celestial (equatorial)')])
    
    if not partial:
        hdr['FIRSTPIX']=odict([('name','FIRSTPIX'),
                               ('value',0), 
                               ('comment','First pixel # (0 based)')])
        hdr['LASTPIX']=odict([('name','LASTPIX'),
                              ('value',hp.nside2npix(nside)-1),
                              ('comment','Last pixel # (0 based)')])
    hdr['INDXSCHM']=odict([('name','INDXSCHM'),
                           ('value','EXPLICIT' if partial else 'IMPLICIT'),
                           ('comment','Indexing: IMPLICIT or EXPLICIT')])
    hdr['OBJECT']=odict([('name','OBJECT'), 
                         ('value','PARTIAL' if partial else 'FULLSKY'),
                         ('comment','Sky coverage, either FULLSKY or PARTIAL')])
    return hdr

def write_partial_map(filename, data, nside, coord=None, ordering='RING',
                      header=None,dtype=None,**kwargs):
    """
    Partial HEALPix maps are used to efficiently store maps of the sky by only
    writing out the pixels that contain data.

    Three-dimensional data can be saved by supplying a distance modulus array
    which is stored in a separate extension.

    Parameters:
    -----------
    filename : output file name
    data     : dictionary or recarray of data to write (must contain 'PIXEL')
    nside    : healpix nside of data
    coord    : 'G'alactic, 'C'elestial, 'E'cliptic
    ordering : 'RING' or 'NEST'
    kwargs   : Passed to fitsio.write

    Returns:
    --------
    None
    """
    # ADW: Do we want to make everything uppercase?

    if isinstance(data,dict):
        names = data.keys()
    else:
        names = data.dtype.names

    if 'PIXEL' not in names:
        msg = "'PIXEL' column not found."
        raise ValueError(msg)

    hdr = header_odict(nside=nside,coord=coord,ordering=ordering)
    fitshdr = fitsio.FITSHDR(hdr.values())
    if header is not None:
        for k,v in header.items():
            fitshdr.add_record({'name':k,'value':v})

    logger.info("Writing %s"%filename)
    fitsio.write(filename,data,extname='PIX_DATA',header=fitshdr,clobber=True)

def read_partial_map(filename, column, fullsky=True, **kwargs):
    """
    Read a partial HEALPix file and return pixels and values/map. Can
    handle 3D healpix maps (pix, value, zdim). Returned array has
    shape (dimz,npix).

    Parameters:
    -----------
    filenames     : list of input filenames
    column        : column of interest
    fullsky       : partial or fullsky map
    kwargs        : passed to fitsio.read

    Returns:
    --------
    (nside,pix,map) : pixel array and healpix map (partial or fullsky)
    """
    # Make sure that PIXEL is in columns
    kwargs['columns'] = ['PIXEL',column]

    filenames = np.atleast_1d(filename)
    header = fitsio.read_header(filenames[0],ext=kwargs.get('ext',1))
    data = ugali.utils.fileio.load_files(filenames,**kwargs)
    #data,header = fitsio.read(filename,**kwargs)

    pix = data['PIXEL']
    value = data[column]
    nside = header['NSIDE']
    npix = hp.nside2npix(nside)

    ndupes = len(pix) - len(np.unique(pix))
    if ndupes > 0:
        msg = '%i duplicate pixels during load.'%(ndupes)
        raise Exception(msg)

    if fullsky:
        shape = list(value.shape)
        shape[0] = npix
        hpxmap = hp.UNSEEN * np.ones(shape,dtype=value.dtype)
        hpxmap[pix] = value
        return (nside,pix,hpxmap.T)
    else:
        return (nside,pix,value.T)

def merge_partial_maps(filenames,outfile,**kwargs):
    filenames = np.atleast_1d(filenames)

    header = fitsio.read_header(filenames[0],ext=kwargs.get('ext',1))
    nside = header['NSIDE']
    data = ugali.utils.fileio.load_files(filenames,**kwargs)
    pix = data['PIXEL']

    ndupes = len(pix) - len(np.unique(pix))
    if ndupes > 0:
        msg = '%i duplicate pixels during load.'%(ndupes)
        raise Exception(msg)

    extname = 'DISTANCE_MODULUS'
    distance = ugali.utils.fileio.load_files(filenames,ext=extname)[extname]
    unique_distance = np.unique(distance)
    # Check if distance moduli are the same...
    if np.any(distance[:len(unique_distance)] != unique_distance):
        msg = "Non-matching distance modulus:"
        msg += '\n'+str(distance[:len(unique_distance)])
        msg += '\n'+str(unique_distance)
        raise Exception(msg)

    write_partial_map(outfile,data=data,nside=nside,clobber=True)
    fitsio.write(outfile,{extname:unique_distance},extname=extname)

def merge_likelihood_headers(filenames, outfile):
    """
    Merge header information from likelihood files.

    Parameters:
    -----------
    filenames : input filenames
    oufile    : the merged file to write
    
    Returns:
    --------
    data      : the data being written
    """

    filenames = np.atleast_1d(filenames)

    ext='PIX_DATA'
    nside = fitsio.read_header(filenames[0],ext=ext)['LKDNSIDE']
    
    keys=['STELLAR','NINSIDE','NANNULUS']
    data_dict = odict(PIXEL=[])
    for k in keys:
        data_dict[k] = []

    for i,filename in enumerate(filenames):
        logger.debug('(%i/%i) %s'%(i+1, len(filenames), filename))
        header = fitsio.read_header(filename,ext=ext)
        data_dict['PIXEL'].append(header['LKDPIX'])
        for key in keys:
            data_dict[key].append(header[key])

        del header
        
    data_dict['PIXEL'] = np.array(data_dict['PIXEL'],dtype=int)
    for key in keys:
        data_dict[key] = np.array(data_dict[key],dtype='f4')

    #import pdb; pdb.set_trace()
    write_partial_map(outfile, data_dict, nside)
    return data_dict

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args
