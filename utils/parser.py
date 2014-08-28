#!/usr/bin/env python



import argparse
import numpy as np

from ugali.utils.healpix import pix2ang
from ugali.utils.projector import cel2gal
from ugali.utils.logger import logger

class Parser(argparse.ArgumentParser):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('formatter_class',
                          argparse.ArgumentDefaultsHelpFormatter)

        super(Parser,self).__init__(*args,**kwargs)

    def add_coords(self,required=False):
        group = self.add_argument_group(title="Coordinates (mutually exclusive)")
        exclusive = group.add_mutually_exclusive_group(required=required)
        exclusive.add_argument('--gal',nargs=2,default=None,metavar=('glon','glat'),
                            type=float,help="Location in Galactic coordinates.")
        exclusive.add_argument('--cel',nargs=2,default=None,metavar=('ra','dec'),
                            type=float,help="Location in celestial coordinates.")
        exclusive.add_argument('--hpx',nargs=2,default=None,metavar=('nside','pix'),
                            type=int,help="Location in HEALPix coordinates.")

    def add_verbose(self,**kwargs):
        self.add_argument('-v','--verbose',action='store_true',
                          help='Output verbosity.',**kwargs)

    def add_debug(self,**kwargs):
        self.add_argument('-d','--debug',action='store_true',
                          help="Setup, but do *NOT* run",**kwargs)

    def add_queue(self,**kwargs):
        self.add_argument('-q','--queue',
                          help="Batch queue for execution.",**kwargs)

    def add_config(self,**kwargs):
        self.add_argument('config',metavar='config.yaml',
                          help='Configuration file (yaml or python dict).',**kwargs)

    def _parse_verbose(self,opts):
        if vars(opts).get('verbose'): 
            logger.setLevel(logger.DEBUG)

    def _parse_coords(self,opts):
        # The coordinates are mutually exclusive, so
        # shouldn't have to worry about over-writing them.
        if 'coords' in vars(opts): return
        if vars(opts).get('gal') is not None: 
            opts.coords = opts.gal
        elif vars(opts).get('cel') is not None: 
            opts.coords = cel2gal(*opts.cel)
        elif vars(opts).get('hpx') is not None: 
            opts.coords = pix2ang(*opts.hpx)
        else:
            opts.coords = None
        
    def parse_args(self,*args,**kwargs):
        opts = super(Parser,self).parse_args(*args,**kwargs)
        self._parse_verbose(opts)
        self._parse_coords(opts)
        return opts

if __name__ == "__main__":
    description = "Argument parser test."
    parser = Parser(description=description)
    parser.add_debug()
    parser.add_verbose()
    parser.add_coords()
    opts = parser.parse_args()
    print opts
