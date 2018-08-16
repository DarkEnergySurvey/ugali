#!/usr/bin/env python
import matplotlib

import healpy
import pylab as plt
import numpy
import astropy.io.fits as pyfits

import ugali.utils.skymap
import ugali.utils.projector
import ugali.utils.plotting
import ugali.utils.config

TITLES = dict(
bootes_I          = "Bootes I"         , 
bootes_II         = "Bootes II"        , 
bootes_III        = "Bootes III"       , 
canes_venatici_I  = "Canes Venatici I" , 
canes_venatici_II = "Canes Venatici II", 
canis_major       = "Canis Major"      , 
carina            = "Carina"           , 
coma_berenices    = "Coma Berenices"   , 
draco             = "Draco"            , 
fornax            = "Fornax"           , 
hercules          = "Hercules"         , 
leo_I             = "Leo I"            , 
leo_II            = "Leo II"           , 
leo_IV            = "Leo IV"           , 
leo_T             = "Leo T"            , 
leo_V             = "Leo V"            , 
lmc               = "LMC"              , 
pisces_II         = "Pisces II"        , 
sagittarius       = "Sagittarius"      , 
sculptor          = "Sculptor"         , 
segue_1           = "Segue 1"          , 
segue_2           = "Segue 2"          , 
sextans           = "Sextans"          , 
smc               = "SMC"              , 
ursa_major_I      = "Ursa Major I"     , 
ursa_major_II     = "Ursa Major II"    , 
ursa_minor        = "Ursa Minor"       , 
willman_1         = "Willman 1"        , 
)
    

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input.fits"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-n','--name',default=None)
    parser.add_option('-t','--targets',default=None)
    parser.add_option('-c','--coord',default='GAL')
    parser.add_option('-p','--proj',default='TAN',choices=['MOL','CAR','TAN'])
    parser.add_option('-f','--field',default='LOG_LIKELIHOOD')

    (opts, args) = parser.parse_args()

    infile = args[0]
    f = pyfits.open(infile)
    nside = f[1].header['NSIDE']
    distance_modulus_array = f[2].data['DISTANCE_MODULUS']

    blank = healpy.UNSEEN * numpy.ones( healpy.nside2npix(nside) )

    pix,richness = ugali.utils.skymap.readSparseHealpixMap(infile,'RICHNESS',construct_map=False)
    pix,frac = ugali.utils.skymap.readSparseHealpixMap(infile,'FRACTION_OBSERVABLE',construct_map=False)
    pix,stellar = ugali.utils.skymap.readSparseHealpixMap(infile,'STELLAR_MASS',construct_map=False)
    pix,loglike = ugali.utils.skymap.readSparseHealpixMap(infile,'LOG_LIKELIHOOD',construct_map=False)
    ts = 2*loglike

    targets = numpy.loadtxt(opts.targets,dtype=[('name',object),('lon',float),
                                                ('lat',float),('radius',float),
                                                ('coord',object),('distance',float)])
    if not targets.shape: targets = targets.reshape(-1)

    for target in targets:
        name = target['name']
        if opts.name is not None and opts.name != name: continue
        try: title = TITLES[name]
        except KeyError: title=name
        distance_modulus = ugali.utils.projector.distanceToDistanceModulus(target['distance'])
        idx = numpy.abs(distance_modulus_array - distance_modulus).argmin()
        print(title, distance_modulus, distance_modulus_array[idx])

        if target['coord'] == 'CEL':
            glon, glat = ugali.utils.projector.celToGal(target['lon'],target['lat'])
        elif target['coord'] == 'GAL':
            glon, glat = target['lon'],target['lat']
        else:
            raise Exception('...')

        plt.ioff()
        fig = plt.figure(figsize=(6,6))
        xsize = 1000
        reso = 60. * 2. * target['radius'] / xsize # Deg to arcmin

        label_kwargs = dict(xy=(0.05,0.05),xycoords='axes fraction', xytext=(0, 0), 
                            textcoords='offset points',ha='left', va='bottom',size=10,
                            bbox={'boxstyle':"round",'fc':'1'}, zorder=10)

        slices = [(ts,'TS'), (stellar,'Stellar'), (frac,'Fraction'), (richness,'Richness')]
        map = blank
        for i,(data,label) in enumerate(slices):
            map[pix] = data[idx] if data.ndim > 1 else data
            img = healpy.gnomview(map, rot=[glon, glat], xsize=xsize, reso=reso,
                            title='',sub=[2,2,i+1],notext=True)
            fig.gca().annotate(label,**label_kwargs)

        fig.suptitle(title,bbox={'boxstyle':"round",'fc':'1'}, zorder=10)
        plt.draw()
        plt.savefig(name+'_results.png',bbox_inches='tight')
