#!/usr/bin/env python
#import mangle
import healpy
import numpy as np
import pylab as plt

from ugali.preprocess.maglims import inMangle

from ugali.utils.projector import pixToAng
if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] mangle1.ply [mangle2.ply ...]"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    (opts, args) = parser.parse_args()

    nside = 2**8
    print("Creating ra, dec...")
    pix = np.arange(healpy.nside2npix(nside))
    ra,dec = pixToAng(nside,pix)

    for infile in args:
        print("Testing %i HEALPix pixels ..."%len(pix))
        inside = inMangle(infile,ra,dec)

        print("Plotting...")
        healpy.mollview(inside)

        outfile = infile.replace('.ply','.png')
        print("Writing %s"%outfile)
        plt.savefig(outfile)
