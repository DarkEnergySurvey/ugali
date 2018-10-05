#!/usr/bin/env python
"""
Working with FITS files.
"""
__author__ = "Alex Drlica-Wagner"

import shutil
import os
from collections import OrderedDict as odict
import itertools
import warnings

import fitsio
import numpy as np
import healpy as hp

from ugali.utils.logger import logger
from ugali.utils.mlab import isstring

def read(filename,**kwargs):
    """ Read a generic input file into a recarray.
    Accepted file formats: [.fits,.fz,.npy,.csv,.txt,.dat]
    
    Parameters:
    filename : input file name
    kwargs   : keyword arguments for the reader
    Returns:
    recarray : data array
    """
    base,ext = os.path.splitext(filename)
    if ext in ('.fits','.fz'):
        # Abstract fits here...
        return fitsio.read(filename,**kwargs)
    elif ext in ('.npy'):
        return np.load(filename,**kwargs)
    elif ext in ('.csv'):
        return np.recfromcsv(filename,**kwargs)
    elif ext in ('.txt','.dat'):
        return np.genfromtxt(filename,**kwargs)

    msg = "Unrecognized file type: %s"%filename
    raise ValueError(msg)

def write(filename,data,**kwargs):
    """ Write a recarray to a specific format.
    Accepted file formats: [.fits,.fz,.npy,.csv,.txt,.dat]
    
    Parameters:
    filename : output file name
    data     : the recarray data
    kwargs   : keyword arguments for the writer
    Returns:
    ret      : writer return (usually None)
    """
    base,ext = os.path.splitext(filename)
    if ext in ('.fits','.fz'):
        # Abstract fits here...
        return fitsio.write(filename,data,**kwargs)
    elif ext in ('.npy'):
        return np.save(filename,data,**kwargs)
    elif ext in ('.csv'):
        return np.savetxt(filename,data,header=','.join(data.dtype.names),delimiter=',',**kwargs)
    elif ext in ('.txt','.dat'):
        return np.savetxt(filename,data,**kwargs)

    msg = "Unrecognized file type: %s"%filename
    raise ValueError(msg)
    

def add_column(filename,column,formula,force=False):
    """ Add a column to a FITS file.

    ADW: Could this be replaced by a ftool?
    """
    columns = parse_formula(formula)
    logger.info("Running file: %s"%filename)
    logger.debug("  Reading columns: %s"%columns)
    data = fitsio.read(filename,columns=columns)

    logger.debug('  Evaluating formula: %s'%formula)
    col = eval(formula)

    col = np.asarray(col,dtype=[(column,col.dtype)])
    insert_columns(filename,col,force=force)
    return True

def load_file(kwargs):
    """ Load a FITS file with kwargs. 

    Parameters:
    kwargs : keyword arguments passed to fitsio.read
    Returns:
    ndarray : fits catalog data
    """
    logger.debug("Loading %s..."%kwargs['filename'])
    return fitsio.read(**kwargs)

def load_files(filenames,multiproc=False,**kwargs):
    """ Load a set of FITS files with kwargs. """
    filenames = np.atleast_1d(filenames)
    logger.debug("Loading %s files..."%len(filenames))

    kwargs = [dict(filename=f,**kwargs) for f in filenames]

    if multiproc:
        from multiprocessing import Pool
        processes = multiproc if multiproc > 0 else None
        p = Pool(processes,maxtasksperchild=1)
        out = p.map(load_file,kwargs)
    else:
        out = [load_file(kw) for kw in kwargs]

    dtype = out[0].dtype
    for i,d in enumerate(out):
        if d.dtype != dtype: 
            # ADW: Not really safe...
            logger.warn("Casting input data to same type.")
            out[i] = d.astype(dtype,copy=False)

    logger.debug('Concatenating arrays...')
    return np.concatenate(out)

def load(args):
    infile,columns = args
    logger.debug("Loading %s..."%infile)
    return fitsio.read(infile,columns=columns)

def load_infiles(infiles,columns=None,multiproc=False):
    if isstring(infiles):
        infiles = [infiles]

    logger.debug("Loading %s files..."%len(infiles))

    args = list(zip(infiles,len(infiles)*[columns]))

    if multiproc:
        from multiprocessing import Pool
        processes = multiproc if multiproc > 0 else None
        p = Pool(processes,maxtasksperchild=1)
        out = p.map(load,args)
    else:
        out = [load(arg) for arg in args]

    dtype = out[0].dtype
    for i,d in enumerate(out):
        if d.dtype != dtype: 
            # ADW: Not really safe...
            logger.warn("Casting input data to same type.")
            out[i] = d.astype(dtype)

    logger.debug('Concatenating arrays...')
    return np.concatenate(out)

def insert_columns(filename,data,ext=1,force=False,colnum=None):
    #logger.info(filename)
    if not os.path.exists(filename):
        msg = "Requested file does not exist."
        raise IOError(msg)

    fits = fitsio.FITS(filename,'rw')
    names = fits[ext].get_colnames()
    overlap = np.in1d(data.dtype.names,names)

    test = None
    if np.any(~overlap):
        idx = np.argmax(np.in1d(names,data.dtype.names))
        test = names[idx]
        orig = fits[ext].read(columns=[test])

    if np.any(overlap) and not force:
        logger.warning("Found overlapping columns; skipping...")
        return
    if len(data) != fits[ext].get_nrows():
        logger.warning("Number of rows does not match; skipping...")
        return
    for col in data.dtype.names:
        if col not in names:
            msg = "Inserting column: %s"%col
            logger.info(msg)
            fits[ext].insert_column(col,data[col],colnum=colnum)
        else:
            msg = "Found column %s"%col
            logger.warning(msg)
            fits[ext].write_column(col,data[col])
        if colnum is not None: colnum += 1

    fits.close()

    # It's already too late since the file has been written...
    if test is not None:
        new = fitsio.read(filename,ext=ext,columns=[test])
        if np.any(new != orig):
            msg = "Input and output do not match!"
            raise Exception(msg)

# Dealing with FITS files
def write_fits(filename,data,header=None,force=False):
    if os.path.exists(filename) and not force:
        found(filename)
        return
    fitsio.write(filename,data,header=header,clobber=force)

# Writing membership files
def write_membership(loglike,filename):
    """
    Write a catalog file of the likelihood region including
    membership properties.

    Parameters:
    -----------
    loglike : input loglikelihood object
    filename : output filename
    
    Returns:
    --------
    None
    """

    ra,dec = gal2cel(loglike.catalog.lon,loglike.catalog.lat)
        
    name_objid = loglike.config['catalog']['objid_field']
    name_mag_1 = loglike.config['catalog']['mag_1_field']
    name_mag_2 = loglike.config['catalog']['mag_2_field']
    name_mag_err_1 = loglike.config['catalog']['mag_err_1_field']
    name_mag_err_2 = loglike.config['catalog']['mag_err_2_field']

    # Angular and isochrone separations
    sep = angsep(loglike.source.lon,loglike.source.lat,
                 loglike.catalog.lon,loglike.catalog.lat)
    isosep = loglike.isochrone.separation(loglike.catalog.mag_1,loglike.catalog.mag_2)

    data = odict()
    data[name_objid] = loglike.catalog.objid
    data['GLON'] = loglike.catalog.lon
    data['GLAT'] = loglike.catalog.lat
    data['RA']   = ra
    data['DEC']  = dec
    data[name_mag_1] = loglike.catalog.mag_1
    data[name_mag_err_1] = loglike.catalog.mag_err_1
    data[name_mag_2] = loglike.catalog.mag_2
    data[name_mag_err_2] = loglike.catalog.mag_err_2
    data['COLOR'] = loglike.catalog.color
    data['ANGSEP'] = sep
    data['ISOSEP'] = isosep
    data['PROB'] = loglike.p

    # HIERARCH allows header keywords longer than 8 characters
    header = []
    for param,value in loglike.source.params.items():
        card = dict(name='HIERARCH %s'%param.upper(),
                    value=value.value,
                    comment=param)
        header.append(card)
    card = dict(name='HIERARCH %s'%'TS',value=loglike.ts(),
                comment='test statistic')
    header.append(card)
    card = dict(name='HIERARCH %s'%'TIMESTAMP',value=time.asctime(),
                comment='creation time')
    header.append(card)
    fitsio.write(filename,data,header=header,clobber=True)
