#!/usr/bin/env python
"""
Script for downloading isochrone grids.
"""
__author__ = "Alex Drlica-Wagner"
import os
import re
import time
import subprocess
from multiprocessing import Pool
from collections import OrderedDict as odict
import copy

import numpy as np

from ugali.utils.logger import logger
from ugali.utils.shell import mkdir
from ugali.isochrone import factory as isochrone_factory


def get_bands(kind, survey):
    """ Bands of a photometric system, for the requested isochrone class.

    The band names depend on the class as well as the survey (PARSEC and MIST
    cover the same filter systems but name and number the bands differently),
    so this reads `band_names` off the class rather than assuming one of the
    module-level tables.
    """
    import ugali.isochrone
    for name in dir(ugali.isochrone):
        cls = getattr(ugali.isochrone, name)
        if not isinstance(cls, type): continue
        if name.lower() != kind.lower(): continue
        return getattr(cls, 'band_names', {}).get(survey.lower())
    return None


# One isochrone instance per (kind, survey) per process. Building it parses a
# file, so it is worth doing once, but it cannot be shared across processes.
_ISOCHRONES = {}


def get_isochrone(kind, survey):
    """ Isochrone instance for this kind and survey, built once per process. """
    key = (kind, survey)
    if key not in _ISOCHRONES:
        # The default bands ('g','r') do not exist in every photometric
        # system, and the isochrone is parsed as soon as it is constructed,
        # so ask for bands that the requested survey actually has.
        kwargs = dict(survey=survey)
        bands = get_bands(kind, survey)
        if bands: kwargs.update(band_1=bands[0], band_2=bands[1])
        _ISOCHRONES[key] = isochrone_factory(kind, **kwargs)
    return _ISOCHRONES[key]


def run(task):
    """ Download a single grid point.

    NOTE: this has to be a module-level function that carries everything it
    needs in its argument, rather than a closure over the isochrone built in
    __main__. Python 3.14 made 'forkserver' the default multiprocessing start
    method on Linux, and a forkserver worker imports the script as
    '__mp_main__', so anything defined inside the __main__ guard does not
    exist in the child -- pool.map then hangs indefinitely instead of failing.
    """
    kind, survey, age, metallicity, outdir, force, delay = task
    try:
        get_isochrone(kind, survey).download(age, metallicity, outdir, force)
        return True
    except Exception as e:
        logger.warning(str(e))
        logger.error("Download failed.")
        return False
    finally:
        # Pace the requests whether or not this one worked: a server that is
        # failing is exactly the one that should not be asked again straight
        # away.
        if delay: time.sleep(delay)


if __name__ == "__main__":
    import ugali.utils.parser
    description = "Download isochrones"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_verbose()
    parser.add_force()
    parser.add_argument('-a','--age',default=None,type=float,action='append')
    parser.add_argument('-z','--metallicity',default=None,type=float,action='append')
    parser.add_argument('-k','--kind',default='Marigo2017')
    parser.add_argument('-s','--survey',default='lsst')
    parser.add_argument('-o','--outdir',default=None)
    parser.add_argument('-n','--njobs',default=1,type=int)
    parser.add_argument('--zmax',default=None,type=float,
                        help='extend the metallicity grid out to this Z')
    parser.add_argument('--dz',default=1e-4,type=float,
                        help='step size of the extended metallicity grid')
    parser.add_argument('--delay',default=0.0,type=float,
                        help='seconds to wait after each download; use this '
                        'to stay under what a server tolerates (mist.science '
                        'degrades under sustained querying)')
    args = parser.parse_args()

    if args.verbose:
        try:
            from http.client import HTTPConnection
        except ImportError:
            from httplib import HTTPConnection
        HTTPConnection.debuglevel = 1

    if args.outdir is None:
        args.outdir = os.path.join(args.survey.lower(),args.kind.lower())
    logger.info("Writing to output directory: %s"%args.outdir)

    iso = get_isochrone(args.kind,args.survey)

    # Defaults
    abins = [args.age] if args.age else iso.abins
    zbins = [args.metallicity] if args.metallicity else iso.zbins

    # The default metallicity grid stops at Z = 1e-3 ([Fe/H] ~ -1.2), which is
    # too metal-poor for some systems; '--zmax' extends it (coarsely).
    if args.zmax is not None:
        extra = np.arange(np.max(zbins)+args.dz, args.zmax+args.dz/2, args.dz)
        zbins = np.concatenate([np.atleast_1d(zbins),extra])

    grid = [g.flatten() for g in np.meshgrid(abins,zbins)]
    logger.info("Ages (Gyr):\n  %s"%np.unique(grid[0]))
    logger.info("Metallicities (Z):\n  %s"%np.unique(grid[1]))
    
    arglist = [(args.kind,args.survey,a,z,args.outdir,args.force,args.delay)
               for a,z in zip(*grid)]
    logger.info("Running %s downloads..."%(len(arglist)))

    # Dotter2008 still downloads through a shared temporary path on the
    # server side; Dotter2016 is safe now that each query unpacks into its
    # own temporary directory.
    if args.njobs > 1 and args.kind.lower() == 'dotter2008':
        msg = "Multiprocessing does not work for %s download."%args.kind
        raise Exception(msg)
    elif args.njobs > 1:
        pool = Pool(processes=args.njobs, maxtasksperchild=100)
        results = pool.map(run,arglist)
    else:
        results = list(map(run,arglist))

    results = np.array(results)
    print("Number of attempted jobs: %s"%len(results))
    print("Number of succesful jobs: %s"%np.sum(results))
    print("Number of failed jobs: %s"%np.sum(~results))
    
