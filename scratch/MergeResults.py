#!/usr/bin/env python

import ugali.utils.skymap
from ugali.utils.logger import logger

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] results1.fits results2.fits ... "
    description = "Script for merging multiple results files."
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-o','--outfile',default="merged.fits")
    parser.add_option('-v','--verbose',action='store_true')
    (opts, args) = parser.parse_args()
    if opts.verbose: logger.setLevel(logger.DEBUG)
    else:            logger.setLevel(logger.INFO)

    ugali.utils.skymap.mergeSparseHealpixMaps(args,opts.outfile)
