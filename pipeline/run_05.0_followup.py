#!/usr/bin/env python
import os
import glob
import numpy
import copy
import subprocess

import ugali.data.pixelize
import ugali.data.maglims
import ugali.utils.projector
import ugali.analysis.farm

from ugali.utils.parse_config import Config
from ugali.utils.projector import celToGal
from ugali.utils.shell import mkdir
from ugali.utils.logger import logger

COMPONENTS = []
if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-t','--targets', default=None)
    parser.add_option('-r','--run', default=[],
                      action='append',choices=COMPONENTS,
                      help="Choose analysis component to run")
    parser.add_option('-v','--verbose', action='store_true')
    parser.add_option('--debug', action='store_true')
    parser.add_option('--local', action='store_true')
    (opts, args) = parser.parse_args()
    if not opts.run: opts.run = COMPONENTS
    if opts.verbose: logger.setLevel(logger.DEBUG)

    if not opts.targets: 
        logger.error("Targets must be specified")
        parser.print_help()
        raise Exception()

    configfile = args[0]
    config = Config(configfile)

    targets = numpy.genfromtxt(opts.targets,dtype=None)
    if not targets.shape: targets = targets.reshape(-1)

    for target in targets:
        name = target[0]
        target_config = copy.deepcopy(config)
        savedir = target_config.params['output']['savedir_likelihood']
        savedir = mkdir(os.path.join(savedir,name))
        logdir = os.path.join(savedir,'log')

        target_config.params['output']['savedir_likelihood'] = savedir
        target_config.params['output']['logdir_likelihood'] = logdir

        distance_modulus = ugali.utils.projector.distanceToDistanceModulus(target[5])
        target_config.params['likelihood']['distance_modulus_array'] = [round(distance_modulus,2)]

        rhalf = target[6]
        target_config.params['kernel']['params'] = [round(rhalf,2)]
        
        outfile = os.path.join(savedir,"config_%s.py"%name)
        target_config.writeConfig(outfile)

        if   target[4] == 'CEL': glon,glat = celToGal(target[1],target[2])
        elif target[4] == 'GAL': glon,glat = target[1],target[2]
        else: raise Exception('Unrecognized coordinate %s'%target[4])
        radius = target[3]
        coords = [ (glon,glat,radius) ]
        print "\n%s (glon=%.2f,glat=%.2f,radius=%.1f)\n"%(name.upper(),glon,glat,radius)
        farm = ugali.analysis.farm.Farm(target_config)
        farm.submit_all(coords=coords,local=opts.local,debug=opts.debug)

