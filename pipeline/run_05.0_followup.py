#!/usr/bin/env python
import os
import glob
import numpy
import copy
import subprocess

import ugali.utils.projector
import ugali.analysis.farm

from ugali.utils.config import Config
from ugali.utils.projector import celToGal
from ugali.utils.shell import mkdir
from ugali.utils.logger import logger

COMPONENTS = []
if __name__ == "__main__":
    import argparse
    description = "Pipeline script for targetted followup"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config',help='Configuration file.')
    parser.add_argument('-t','--targets', default=None, required=True)
    parser.add_argument('-v','--verbose',action='store_true',
                        help='Output verbosity')
    parser.add_argument('-r','--run', default=[],
                        action='append',choices=COMPONENTS,
                        help="Choose analysis component to run")
    parser.add_arguments('--debug', action='store_true')
    parser.add_arguments('--local', action='store_true')
    opts = parser.parse_args()

    config = Config(opts.config)
    if not opts.run: opts.run = COMPONENTS
    if opts.verbose: logger.setLevel(logger.DEBUG)

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

