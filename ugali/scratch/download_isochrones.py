#!/usr/bin/env python
"""
Script for downloading isochrone grids.
"""
__author__ = "Alex Drlica-Wagner"
import os
from multiprocessing import Pool

import numpy as np

from ugali.utils.logger import logger
from ugali.isochrone import factory as isochrone_factory

if __name__ == "__main__":
    import ugali.utils.parser
    description = "Download isochrones"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_verbose()
    parser.add_force()
    parser.add_argument('-a','--age',default=None,type=float,action='append')
    parser.add_argument('-z','--metallicity',default=None,type=float,action='append')
    parser.add_argument('-k','--kind',default='Marigo2017')
    parser.add_argument('-s','--survey',default='lsst_dp0')
    parser.add_argument('-o','--outdir',default=None)
    parser.add_argument('-n','--njobs',default=1,type=int)
    args = parser.parse_args()

    if args.verbose:
        from httplib import HTTPConnection
        HTTPConnection.debuglevel = 1

    if args.outdir is None:
        args.outdir = os.path.join(args.survey.lower(),args.kind.lower())
    logger.info("Writing to output directory: %s"%args.outdir)

    iso = isochrone_factory(args.kind,survey=args.survey)

    # Defaults
    abins = [args.age] if args.age else iso.abins
    zbins = [args.metallicity] if args.metallicity else iso.zbins
    grid = [g.flatten() for g in np.meshgrid(abins,zbins)]
    logger.info("Ages (Gyr):\n  %s"%np.unique(grid[0]))
    logger.info("Metallicities (Z):\n  %s"%np.unique(grid[1]))
    
    def run(args):
        try:
            iso.download(*args)
            return True
        except Exception as e:
            logger.warn(str(e))
            logger.error("Download failed.")
            return False

    arglist = [(a,z,args.outdir,args.force) for a,z in zip(*grid)]
    logger.info("Running %s downloads..."%(len(arglist)))

    if args.njobs > 1 and args.kind.startswith('Dotter'):
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
    
