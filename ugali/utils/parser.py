#!/usr/bin/env python
"""

@author: Alex Drlica-Wagner <kadrlica@fnal.gov>
"""
import os
import argparse

import numpy as np
import numpy.lib.recfunctions as recfuncs
import pyfits

from ugali.utils.healpix import pix2ang
from ugali.utils.projector import cel2gal
from ugali.utils.logger import logger

class Parser(argparse.ArgumentParser):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('formatter_class',
                          argparse.ArgumentDefaultsHelpFormatter)

        super(Parser,self).__init__(*args,**kwargs)


    def add_coords(self,required=False,radius=False,targets=False):
        group = self.add_argument_group(title="target coordinates")
        exclusive = group.add_mutually_exclusive_group(required=required)
        if targets:
            exclusive.add_argument('-t','--targets',default=None,metavar='targets.txt',
                                   help="List of target coordinates")
        exclusive.add_argument('--gal',nargs=2,default=None,metavar=('glon','glat'),
                            type=float,help="Location in Galactic coordinates.")
        exclusive.add_argument('--cel',nargs=2,default=None,metavar=('ra','dec'),
                            type=float,help="Location in celestial coordinates.")
        exclusive.add_argument('--hpx',nargs=2,default=None,metavar=('nside','pix'),
                            type=int,help="Location in HEALPix coordinates.")
        if radius:
            group.add_argument('--radius',default=0,type=float,
                               help="Radius surrounding specified coordinates")

    def add_name(self,**kwargs):
        self.add_argument('-n','--name',
                          help = "Target name",**kwargs)

    def add_verbose(self,**kwargs):
        self.add_argument('-v','--verbose',action='store_true',
                          help='Output verbosity.',**kwargs)

    def add_debug(self,**kwargs):
        self.add_argument('-d','--debug',action='store_true',
                          help="Setup, but do not run",**kwargs)

    def add_queue(self,**kwargs):
        self.add_argument('-q','--queue',
                          help="Batch queue for execution.",**kwargs)
                          
    def add_ncores(self,**kwargs):
        self.add_argument('--ncores',
                          help="Number of cores to use.",**kwargs)

    def add_config(self,**kwargs):
        self.add_argument('config',metavar='config.yaml',
                          help='Configuration file (yaml or python dict).',**kwargs)


    def add_force(self,**kwargs):
        self.add_argument('-f','--force',action='store_true',
                          help='Force the overwrite of files',**kwargs)

    def add_seed(self,**kwargs):
        self.add_argument('--seed',default=None,type=int,
                          help='Random seed.',**kwargs)

    def add_version(self,**kwargs):
        from ugali import __version__
        self.add_argument('-V','--version',action='version',
                          version='ugali '+__version__,
                          help='Print version.',**kwargs)

    def add_run(self,**kwargs):
        self.add_argument('-r','--run', default=[], action='append',
                          help="Analysis component(s) to run.", **kwargs)

    def _parse_verbose(self,opts):
        if vars(opts).get('verbose') or vars(opts).get('debug'): 
            logger.setLevel(logger.DEBUG)

    def _parse_coords(self,opts):
        # The coordinates are mutually exclusive, so
        # shouldn't have to worry about over-writing them.
        if 'coords' in vars(opts): return
        radius = vars(opts).get('radius',0)
        gal = None
        if vars(opts).get('gal') is not None: 
            gal = opts.gal
        elif vars(opts).get('cel') is not None: 
            gal = cel2gal(*opts.cel)
        elif vars(opts).get('hpx') is not None: 
            gal = pix2ang(*opts.hpx)

        if gal is not None:
            opts.coords = [(gal[0],gal[1],radius)]
            opts.names = [vars(opts).get('name','')]
        else:
            opts.coords = None
            opts.names = None

        if vars(opts).get('targets') is not None:
            opts.names,opts.coords = self.parse_targets(opts.targets)
            if vars(opts).get('radius') is not None:
                opts.coords[:,2] = vars(opts).get('radius')
            
    @staticmethod
    def parse_targets(filename):
        """
        Load a text file with target coordinates. Returns
        an array of target locations in Galactic coordinates.
        File description:
        [NAME] [LON] [LAT] [RADIUS] [COORD]
        
        The values of LON and LAT will depend on COORD:
        COORD = [GAL  | CEL | HPX  ],
        LON   = [GLON | RA  | NSIDE]
        LAT   = [GLAT | DEC | PIX  ]

        """
        base,ext = os.path.splitext(filename)
        if ext == '.fits':
            f = pyfits.open(filename)
            data = f[1].data.view(np.recarray)
            data = recfuncs.append_fields(data,'RADIUS',np.zeros(len(data)),usemask=False)
            return data['NAME'],data[['GLON','GLAT','RADIUS']]
        elif (ext=='.txt') or (ext=='.dat'):
            data = np.loadtxt(filename,unpack=True,usecols=range(5),dtype=object)
            # Deal with one-line input files
            if data.ndim == 1: data = np.array([data]).T
            names = data[0]
            out   = data[1:4].astype(float)
            lon,lat,radius = out
             
            coord = np.array([s.lower() for s in data[4]])
            gal = (coord=='gal')
            cel = (coord=='cel')
            hpx = (coord=='hpx')
             
            if cel.any():
                glon,glat = cel2gal(lon[cel],lat[cel])
                out[0][cel] = glon
                out[1][cel] = glat
            if hpx.any():
                glon,glat = pix2ang(lat[hpx],lon[hpx])
                out[0][hpx] = glon
                out[1][hpx] = glat
             
            return names,out.T

    #def _parse_local(self,opts):
    #    if vars(opts).get('local') is not None: return
    #    opts.local = (vars(opts).get('queue')=='local')

    def parse_args(self,*args,**kwargs):
        opts = super(Parser,self).parse_args(*args,**kwargs)
        self._parse_verbose(opts)
        self._parse_coords(opts)
        #self._parse_local(opts)
        return opts

if __name__ == "__main__":
    description = "Argument parser test."
    parser = Parser(description=description)
    parser.add_debug()
    parser.add_verbose()
    parser.add_version()
    parser.add_coords()
    opts = parser.parse_args()
    print opts
