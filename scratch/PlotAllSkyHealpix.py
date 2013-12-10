#!/usr/bin/env python
import healpy
import pylab as plt
import ugali.utils.skymap
from ugali.utils.projector import celToGal
import numpy

default_kwargs = dict( xytext=(5,5),textcoords='offset points',
                       ha="left",va="center",
                       color='w', size=8, weight='bold', 
                   )

TARGETS = dict(
bootes_I          = ["Boo I"   ,dict(default_kwargs,xytext=(2,7)),             ], 
bootes_II         = ["Boo II"  ,dict(default_kwargs,xytext=(5,0)),             ], 
bootes_III        = ["Boo III" ,dict(default_kwargs,xytext=(3,-8),ha='right'), ], 
canes_venatici_I  = ["CVn I"   ,dict(default_kwargs,xytext=(-3,-6),ha='right'),], 
canes_venatici_II = ["CVn II"  ,dict(default_kwargs,xytext=(-4,-2),ha='right'),], 
canis_major       = ["CMa"     ,default_kwargs,                                ], 
carina            = ["Car"     ,default_kwargs,                                ], 
coma_berenices    = ["Com"     ,dict(default_kwargs,xytext=(5,-1)),            ], 
draco             = ["Dra"     ,default_kwargs,                                ], 
fornax            = ["For"     ,default_kwargs,                                ], 
hercules          = ["Her"     ,default_kwargs,                                ], 
leo_I             = ["Leo I"   ,dict(default_kwargs,xytext=(4,-3),ha='left'),  ], 
leo_II            = ["Leo II"  ,dict(default_kwargs,xytext=(-4,0),ha='right'), ], 
leo_IV            = ["Leo IV"  ,dict(default_kwargs,xytext=(5,-1),ha='left'),  ], 
leo_T             = ["Leo T"   ,dict(default_kwargs,xytext=(4,5),ha='left'),   ], 
leo_V             = ["Leo V"   ,dict(default_kwargs,xytext=(4,5),ha='left'),   ], 
lmc               = ["LMC"     ,default_kwargs,                                ], 
pisces_II         = ["Psc II"  ,default_kwargs,                                ], 
sagittarius       = ["Sgr"     ,default_kwargs,                                ], 
sculptor          = ["Scl"     ,default_kwargs,                                ], 
segue_1           = ["Seg 1"   ,dict(default_kwargs,xytext=(4,2),ha='left'),   ], 
segue_2           = ["Seg 2"   ,default_kwargs,                                ], 
sextans           = ["Sex"     ,dict(default_kwargs,xytext=(-2,-5),ha='right'),], 
smc               = ["SMC"     ,default_kwargs,                                ], 
ursa_major_I      = ["UMa I"   ,dict(default_kwargs,xytext=(5,-5),ha='left'),  ], 
ursa_major_II     = ["UMa II"  ,dict(default_kwargs,xytext=(4,4)),             ], 
ursa_minor        = ["UMi"     ,default_kwargs,                                ], 
willman_1         = ["Wil 1"   ,dict(default_kwargs,xytext=(5,1),ha='left'),   ], 
)


if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-o','--outfile',default='allsky_maglims.png')
    parser.add_option('-t','--targets',default=None)
    parser.add_option('-c','--coord',default='GAL')
    parser.add_option('-p','--proj',default='MOL',choices=['MOL','CAR'])
    parser.add_option('-f','--field',default='MAGLIM')
    parser.add_option('-v','--verbose', action='store_true')
    (opts, args) = parser.parse_args()
    if opts.verbose: logger.setLevel(logger.DEBUG)

    map = ugali.utils.skymap.readSparseHealpixMaps(args,opts.field)
    if opts.coord.upper() == "GAL":
        coord = 'G'
    elif opts.coord.upper() == "CEL":
        coord = 'GC'
    if opts.proj.upper() == "MOL":
        #map = numpy.where( map < 20, healpy.UNSEEN, map)
        healpy.mollview(map,coord=coord,xsize=1000,min=20)
    elif opts.proj.upper() == "CAR":
        healpy.cartview(map,coord=coord,xsize=1000)
    else:
        raise Exception("...")
    healpy.graticule()

    if opts.targets:
        targets = numpy.genfromtxt(opts.targets,unpack=True,dtype=None)
        if not targets.shape: targets = targets.reshape(-1)
        coord = 'CG' # For RA/DEC input
        healpy.projscatter(targets['f1'],targets['f2'],
                           lonlat=True,coord=coord,marker='o',c='w')
        fig = plt.gcf()
        # This is pretty hackish (but is how healpy does it...)
        for ax in fig.get_axes():
            if isinstance(ax,healpy.projaxes.SphericalProjAxes): break

        for target in targets:
            text = TARGETS[target[0]][0]
            kwargs = TARGETS[target[0]][1]
            glon,glat = celToGal(target[1],target[2])


            vec = healpy.rotator.dir2vec(glon,glat,lonlat=True)
            vec = (healpy.rotator.Rotator(rot=None,coord='G',eulertype='Y')).I(vec)

            x,y = ax.proj.vec2xy(vec,direct=False)
            ax.annotate(text,xy=(x,y),xycoords='data',**kwargs)

    plt.savefig(opts.outfile)
