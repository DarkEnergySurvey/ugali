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

import ugali.utils.projector
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
    Return the indices of sub-pixels (resolution nside_subpix) within
    the super-pixel with (resolution nside_superpix).
    
    ADW: It would be better to convert to next and do this explicitly
    """
    if nside_superpix==nside_subpix: return superpix
    vec = hp.pix2vec(nside_superpix, superpix)
    radius = np.degrees(2. * hp.max_pixrad(nside_superpix))
    subpix = query_disc(nside_subpix, vec, radius)
    pix_for_subpix = superpixel(subpix,nside_subpix,nside_superpix)
    # Might be able to speed up array indexing...
    return subpix[pix_for_subpix == superpix]

def d_grade_ipix(ipix, nside_in, nside_out, nest=False):
    """
    Return the indices of the super-pixels which contain each of the
    sub-pixels (nside_in > nside_out).

    Parameters:
    -----------
    ipix      : index of the input subpixels
    nside_in  : nside of the input subpix
    nside_out : nside of the desired superpixels

    Returns:
    --------
    ipix_out  : superpixels for each subpixel
    """

    if nside_in==nside_out: return ipix
    if not (nside_in > nside_out): 
        raise ValueError("nside_out must be less than nside_in")

    return hp.vec2pix(nside_out, *hp.pix2vec(nside_in, ipix, nest), nest=nest)

def u_grade_ipix(ipix, nside_in, nside_out, nest=False):
    """
    Return the indices of sub-pixels (resolution nside_subpix) within
    the super-pixel(s) (resolution nside_superpix).
    
    Parameters:
    -----------
    ipix      : index of the input superpixel(s)
    nside_in  : nside of the input superpixel
    nside_out : nside of the desired subpixels

    Returns:
    --------
    ipix_out : subpixels for each superpixel
    """

    if nside_in==nside_out: return ipix
    if not (nside_in < nside_out): 
        raise ValueError("nside_in must be less than nside_out")

    if nest: nest_ipix = ipix
    else:    nest_ipix = hp.ring2nest(nside_in, ipix)

    factor = (nside_out//nside_in)**2
    if np.isscalar(ipix):
        nest_ipix_out = factor*nest_ipix + np.arange(factor)
    else:
        nest_ipix_out = factor*np.asarray(nest_ipix)[:,np.newaxis]+np.arange(factor)

    if nest: return nest_ipix_out
    else:    return hp.nest2ring(nside_out, nest_ipix_out)
        

def ud_grade_ipix(ipix, nside_in, nside_out, nest=False):
    """
    Upgrade or degrade resolution of a pixel list.

    Parameters:
    -----------
    ipix:array-like 
    the input pixel(s)

    nside_in:int
    the nside of the input pixel(s)

    nside_out:int
    the desired nside of the output pixel(s)

    order:str
    pixel ordering of input and output ("RING" or "NESTED")

    Returns:
    --------
    pix_out:array-like
    the upgraded or degraded pixel array
    """
    if nside_in == nside_out: return ipix
    elif nside_in < nside_out:
        return u_grade_ipix(ipix, nside_in, nside_out, nest)
    elif nside_in > nside_out:
        return d_grade_ipix(ipix, nside_in, nside_out, nest)

############################################################

#ADW: These can be replaced by healpy functions...

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

def get_nside(m):
    try: 
        return hp.get_nside(m)
    except TypeError:
        return hp.get_nside(m.data)

############################################################

def healpixMap(nside, lon, lat, fill_value=0., nest=False):
    """
    Input (lon, lat) in degrees instead of (theta, phi) in radians.
    Returns HEALPix map at the desired resolution 
    """

    lon_median, lat_median = np.median(lon), np.median(lat)
    max_angsep = np.max(ugali.utils.projector.angsep(lon, lat, lon_median, lat_median))
    
    pix = angToPix(nside, lon, lat, nest=nest)
    if max_angsep < 10:
        # More efficient histograming for small regions of sky
        m = np.tile(fill_value, healpy.nside2npix(nside))
        pix_subset = ugali.utils.healpix.angToDisc(nside, lon_median, lat_median, max_angsep, nest=nest)
        bins = np.arange(np.min(pix_subset), np.max(pix_subset) + 1)
        m_subset = np.histogram(pix, bins=bins - 0.5)[0].astype(float)
        m[bins[0:-1]] = m_subset
    else:
        m = np.histogram(pix, np.arange(hp.nside2npix(nside) + 1))[0].astype(float)
    if fill_value != 0.:
        m[m == 0.] = fill_value
    return m

############################################################

def in_pixels(lon,lat,pixels,nside):
    """
    Check if (lon,lat) in pixel list. Assumes RING formatting.

    Parameters:
    -----------
    lon    : longitude (deg)
    lat    : latitude (deg)
    pixels : pixel list [RING format] to check for inclusion
    nside  : nside of pixel list 

    Returns:
    --------
    inpix : boolean array for inclusion
    """
    pix = ang2pix(nside,lon,lat)
    return np.in1d(pix,pixels)


def index_pix_in_pixels(pix,pixels,sort=False,outside=-1):
    """
    Find the indices of a set of pixels into another set of pixels.
    !!! ASSUMES SORTED PIXELS !!!

    Parameters:
    -----------
    pix    : set of search pixels
    pixels : set of reference pixels
  
    Returns:
    --------
    index : index into the reference pixels
    """
    # ADW: Not really safe to set index = -1 (accesses last entry); 
    # -np.inf would be better, but breaks other code...

    # ADW: Are the pixels always sorted? Is there a quick way to check?
    if sort: pixels = np.sort(pixels)

    # Assumes that 'pixels' is pre-sorted, otherwise...???
    index = np.searchsorted(pixels,pix)
    if np.isscalar(index):
        if not np.in1d(pix,pixels).any(): index = outside
    else:
        # Find objects that are outside the pixels
        index[~np.in1d(pix,pixels)] = outside
    return index

def index_lonlat_in_pixels(lon,lat,pixels,nside,sort=False,outside=-1):
    """
    Find the indices of a set of angles into a set of pixels

    Parameters:
    -----------
    pix    : set of search pixels
    pixels : set of reference pixels
  
    Returns:
    --------
    index : index into the reference pixels
    """

    pix = ang2pix(nside,lon,lat)
    return index_pix_in_pixels(pix,pixels,sort,outside)

index_pixels = index_lonlat_in_pixels

#def index_pixels(lon,lat,pixels,nside):
#    """
#    Find the index for object amoung a subset of healpix pixels.
#    Set index of objects outside the pixel subset to -1
#    """
#    # ADW: Not really safe to set index = -1 (accesses last entry); 
#    # -np.inf would be better, but breaks other code...
#    pix = ang2pix(nside,lon,lat)
#    # pixels should be pre-sorted, otherwise...???
#    index = np.searchsorted(pixels,pix)
#    if np.isscalar(index):
#        if not np.in1d(pix,pixels).any(): index = -1
#    else:
#        # Find objects that are outside the roi
#        #index[np.take(pixels,index,mode='clip')!=pix] = -1
#        index[~np.in1d(pix,pixels)] = -1
#    return index


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
        print(e)
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

def header_odict(nside,nest=False,coord=None, partial=True):
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

def write_partial_map(filename, data, nside, coord=None, nest=False,
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
        names = list(data.keys())
    else:
        names = data.dtype.names

    if 'PIXEL' not in names:
        msg = "'PIXEL' column not found."
        raise ValueError(msg)

    hdr = header_odict(nside=nside,coord=coord,nest=nest)
    fitshdr = fitsio.FITSHDR(list(hdr.values()))
    if header is not None:
        for k,v in header.items():
            fitshdr.add_record({'name':k,'value':v})

    logger.info("Writing %s"%filename)
    fitsio.write(filename,data,extname='PIX_DATA',header=fitshdr,clobber=True)

def read_partial_map(filenames, column, fullsky=True, **kwargs):
    """
    Read a partial HEALPix file(s) and return pixels and values/map. Can
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
    #kwargs['columns'] = ['PIXEL',column]
    kwargs['columns'] = ['PIXEL'] + np.atleast_1d(column).tolist()

    filenames = np.atleast_1d(filenames)
    header = fitsio.read_header(filenames[0],ext=kwargs.get('ext',1))
    data = ugali.utils.fileio.load_files(filenames,**kwargs)

    pix = data['PIXEL']
    value = data[column]
    nside = header['NSIDE']
    npix = hp.nside2npix(nside)

    ndupes = len(pix) - len(np.unique(pix))
    if ndupes > 0:
        msg = '%i duplicate pixels during load.'%(ndupes)
        raise Exception(msg)

    if fullsky and not np.isscalar(column):
        raise Exception("Cannot make fullsky map from list of columns.")
    
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


def read_map(filename, nest=False, hdu=None, h=False, verbose=True):
    """Read a healpix map from a fits file.  Partial-sky files,
    if properly identified, are expanded to full size and filled with UNSEEN.
    Uses fitsio to mirror much (but not all) of the functionality of healpy.read_map
    
    Parameters:
    -----------
    filename : str 
      the fits file name
    nest : bool, optional
      If True return the map in NEST ordering, otherwise in RING ordering;
      use fits keyword ORDERING to decide whether conversion is needed or not
      If None, no conversion is performed.
    hdu : int, optional
      the header number to look at (start at 0)
    h : bool, optional
      If True, return also the header. Default: False.
    verbose : bool, optional
      If True, print a number of diagnostic messages
    
    Returns
    -------
    m [, header] : array, optionally with header appended
      The map read from the file, and the header if *h* is True.
    """
    
    data,hdr = fitsio.read(filename,header=True,ext=hdu)

    nside = int(hdr.get('NSIDE'))
    if verbose: print('NSIDE = {0:d}'.format(nside))

    if not healpy.isnsideok(nside):
        raise ValueError('Wrong nside parameter.')
    sz=healpy.nside2npix(nside)

    ordering = hdr.get('ORDERING','UNDEF').strip()
    if verbose: print('ORDERING = {0:s} in fits file'.format(ordering))

    schm = hdr.get('INDXSCHM', 'UNDEF').strip()
    if verbose: print('INDXSCHM = {0:s}'.format(schm))
    if schm == 'EXPLICIT':
        partial = True
    elif schm == 'IMPLICIT':
        partial = False

    # monkey patch on a field method
    fields = data.dtype.names

    # Could be done more efficiently (but complicated) by reordering first
    if hdr['INDXSCHM'] == 'EXPLICIT':
        m = healpy.UNSEEN*np.ones(sz,dtype=data[fields[1]].dtype)
        m[data[fields[0]]] = data[fields[1]]
    else:
        m = data[fields[0]].ravel()

    if (not healpy.isnpixok(m.size) or (sz>0 and sz != m.size)) and verbose:
        print('nside={0:d}, sz={1:d}, m.size={2:d}'.format(nside,sz,m.size))
        raise ValueError('Wrong nside parameter.')
    if not nest is None:
        if nest and ordering.startswith('RING'):
            idx = healpy.nest2ring(nside,np.arange(m.size,dtype=np.int32))
            if verbose: print('Ordering converted to NEST')
            m = m[idx]
            return  m[idx]
        elif (not nest) and ordering.startswith('NESTED'):
            idx = healpy.ring2nest(nside,np.arange(m.size,dtype=np.int32))
            m = m[idx]
            if verbose: print('Ordering converted to RING')

    if h:
        return m, header
    else:
        return m

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args
