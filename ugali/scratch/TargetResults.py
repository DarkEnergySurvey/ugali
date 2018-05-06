#!/usr/bin/env python

import os
import healpy
import pylab as plt
import numpy
import astropy.io.fits as pyfits
import copy

import ugali.utils.skymap
import ugali.utils.projector
import ugali.utils.plotting
import ugali.utils.config

from ugali.utils.projector import pixToAng,angToVec

TITLES = dict(
bootes_I          = ["Boo I"   ,"Bootes I"         ], 
bootes_II         = ["Boo II"  ,"Bootes II"        ], 
bootes_III        = ["Boo III" ,"Bootes III"       ], 
canes_venatici_I  = ["CVn I"   ,"Canes Venatici I" ], 
canes_venatici_II = ["CVn II"  ,"Canes Venatici II"], 
canis_major       = ["CMa"     ,"Canis Major"      ], 
carina            = ["Car"     ,"Carina"           ], 
coma_berenices    = ["Com"     ,"Coma Berenices"   ], 
draco             = ["Dra"     ,"Draco"            ], 
fornax            = ["For"     ,"Fornax"           ], 
hercules          = ["Her"     ,"Hercules"         ], 
leo_I             = ["Leo I"   ,"Leo I"            ], 
leo_II            = ["Leo II"  ,"Leo II"           ], 
leo_IV            = ["Leo IV"  ,"Leo IV"           ], 
leo_T             = ["Leo T"   ,"Leo T"            ], 
leo_V             = ["Leo V"   ,"Leo V"            ], 
lmc               = ["LMC"     ,"LMC"              ], 
pisces_II         = ["Psc II"  ,"Pisces II"        ], 
sagittarius       = ["Sgr"     ,"Sagittarius"      ], 
sculptor          = ["Scl"     ,"Sculptor"         ], 
segue_1           = ["Seg 1"   ,"Segue 1"          ], 
segue_2           = ["Seg 2"   ,"Segue 2"          ], 
sextans           = ["Sex"     ,"Sextans"          ], 
smc               = ["SMC"     ,"SMC"              ], 
ursa_major_I      = ["UMa I"   ,"Ursa Major I"     ], 
ursa_major_II     = ["UMa II"  ,"Ursa Major II"    ], 
ursa_minor        = ["UMi"     ,"Ursa Minor"       ], 
willman_1         = ["Wil 1"   ,"Willman 1"        ], 
)

MAGVSUN = VMAGSUN = 4.83

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input.fits"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-t','--targets',default=None)
    parser.add_option('-c','--coord',default='GAL')
    parser.add_option('-p','--proj',default='TAN',choices=['MOL','CAR','TAN'])
    parser.add_option('-f','--field',default='LOG_LIKELIHOOD')

    (opts, args) = parser.parse_args()

    targets = numpy.genfromtxt(opts.targets,dtype=None)
    if not targets.shape: targets = targets.reshape(-1)

    outfile = "target_results.txt"
    out = open(outfile,'w')

    header = """{:<23s}{:<8s}{:<8s}{:<8s}{:<8s}{:<8s}{:<8s}{:<8s}{:<8s}{:<8s}{:<8s}{:<8s}\n"""
    out.write(header.format("#Name","RA","DEC","RADIUS","COORD","DISTANCE","RHALF","MAGV","(ERR)","MASS","(ERR)","TS"))
    fmt = """{:<23s}{:<8.2f}{:<8.2f}{:<8.1f}{:<8s}{:<8.0f}{:<8.2f}{:<8.1f}{:<8.1f}{:<8.1e}{:<8.1e}{:<8.1f}\n"""


    for target in targets:
        name = target[0]
        try: title = TITLES[name]
        except KeyError: title=name

        if target[4] == 'CEL':
            glon, glat = ugali.utils.projector.celToGal(float(target[1]),float(target[2]))
        elif target[4] == 'GAL':
            glon, glat = float(target[1]),float(target[2])
        else:
            raise Exception('...')
        print(target[0],"(%.2f,%.2f)"%(glon,glat))

        infile = "%s_merged.fits"%(name)
        if not os.path.exists(infile):
            print("WARNING: %s does not exist; skipping..."%infile)
            continue
        f = pyfits.open(infile)
        data = f[1].data
        nside = f[1].header['NSIDE']
        pix = data['PIX']
        distance_modulus = ugali.utils.projector.distanceToDistanceModulus(float(target[5]))
        distance_modulus_array = f[2].data['DISTANCE_MODULUS']
        if len(distance_modulus_array) == 1:
            ii = slice(None)
        else:
            ii = numpy.abs(distance_modulus_array - distance_modulus).argmin()

        vec = angToVec(glon,glat)
        subpix = ugali.utils.projector.query_disc(nside, vec, 0.1, inclusive=True,fact=32)
        sort_pix_idx = pix.argsort()
        sort_pix = pix[sort_pix_idx]
        sort_subpix_idx = numpy.searchsorted(sort_pix,subpix)
        subpix_idx = sort_pix_idx[sort_subpix_idx]

        pixel = subpix[data['LOG_LIKELIHOOD'][ii][subpix_idx].argmax()] 
        idx = numpy.nonzero(pix == pixel)[0]
        ts = float(2*data['LOG_LIKELIHOOD'][ii][idx])
        mass = float(data['STELLAR_MASS'][ii][idx])
        richness = float(data['RICHNESS'][ii][idx])
        richness_upper = float(data['RICHNESS_UPPER'][ii][idx] )
        richness_lower = float(data['RICHNESS_LOWER'][ii][idx] )
        mass_upper = mass * richness_upper/richness
        mass_lower = mass * richness_lower/richness
        mass_err = numpy.mean([mass_upper-mass,mass-mass_lower])
        glon_max,glat_max = pixToAng(nside,pixel)

        print("\t","(%.2f,%.2f)"%(glon_max,glat_max),"TS = %.1f"%data['LOG_LIKELIHOOD'][ii][idx])
        print("\t","mass = %.2e+/-%.1e"%(mass,mass_err))
        luminosity = mass
        magv = float(MAGVSUN - 2.5*numpy.log10(mass))
        magv_lower = MAGVSUN - 2.5*numpy.log10(mass_upper)
        magv_upper = MAGVSUN - 2.5*numpy.log10(mass_lower)
        magv_err =  numpy.mean([magv_upper-magv,magv-magv_lower])
        print("\tmagv_0 = %.1f+/-%.1f"%(target[7],target[8]))
        print("\tmagv = %.1f+/-%.1f"%(magv,magv_err))
        output = list(target)
        output[7] = magv
        output[8] = magv_err
        output+= [mass, mass_err, ts]
        #print "\t",output
        out.write(fmt.format(*output))
        print()
