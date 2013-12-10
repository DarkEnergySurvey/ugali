#!/usr/bin/env python

import ugali.utils.skymap

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] results1.fits results2.fits ... "
    description = "Script for merging multiple results files."
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-o','--outfile',default="merged.fits")
    (opts, args) = parser.parse_args()

    ugali.utils.skymap.mergeSparseHealpixMaps(args,opts.outfile)
