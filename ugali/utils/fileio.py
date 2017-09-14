#!/usr/bin/env python
"""
Working with FITS files.
"""
__author__ = "Alex Drlica-Wagner"

import shutil
import os
from collections import OrderedDict as odict
import itertools

import fitsio
import numpy as np
import healpy as hp

from ugali.utils.logger import logger
#import logging as logger
import warnings

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
    """ Load a FITS file with kwargs. """
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
    if isinstance(infiles,basestring):
        infiles = [infiles]

    logger.debug("Loading %s files..."%len(infiles))

    args = zip(infiles,len(infiles)*[columns])

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

