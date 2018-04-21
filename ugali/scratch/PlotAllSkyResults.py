#!/usr/bin/env python
import healpy
import pylab as plt
import ugali.utils.skymap
import numpy
import astropy.io.fits as pyfits

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-o','--outfile',default='allsky_results.png')
    parser.add_option('-t','--targets',default=None)
    parser.add_option('-c','--coord',default='GAL')
    parser.add_option('-p','--proj',default='MOL',choices=['MOL','CAR'])
    parser.add_option('-f','--field',default='LOG_LIKELIHOOD')

    (opts, args) = parser.parse_args()

    nside = pyfits.open(args[0])[1].header['NSIDE']
    map = healpy.UNSEEN * numpy.ones( healpy.nside2npix(nside) )
    pix,vals = ugali.utils.skymap.readSparseHealpixMap(args[0],opts.field,construct_map=False)
    map[pix] = vals[0]

    if opts.coord.upper() == "GAL":
        coord = 'G'
    elif opts.coord.upper() == "CEL":
        coord = 'GC'
    if opts.proj.upper() == "MOL":
        healpy.mollview(map,coord=coord,xsize=1000)
    elif opts.proj.upper() == "CAR":
        healpy.cartview(map,coord=coord,xsize=1000)
    else:
        raise Exception("...")
    healpy.graticule()

    if opts.targets:
        data = numpy.loadtxt(opts.targets,unpack=True,dtype='str')
        coord = 'CG' # For RA/DEC input
        healpy.projscatter(data[1].astype(float),data[2].astype(float),
                           lonlat=True,coord=coord,marker='o',c='w')
    plt.savefig(opts.outfile)
