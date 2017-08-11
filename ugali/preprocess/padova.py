#!/usr/bin/env python
"""
Download Padova isochrones from:
http://stev.oapd.inaf.it/cgi-bin/cmd

Adapted from ezpadova by Morgan Fouesneau:
https://github.com/mfouesneau/ezpadova

"""
import os
from urllib import urlencode
from urllib2 import urlopen
from StringIO import StringIO
import zlib
import re
import subprocess
from multiprocessing import Pool
from collections import OrderedDict as odict

import numpy as np
from ugali.utils.logger import logger
from ugali.utils.shell import mkdir
from ugali.analysis.isochrone import PadovaIsochrone,OldPadovaIsochrone

# Commented options may need to be restored for older version/isochrones.
# The parameters were tracked down by:
# Chrome -> View -> Developer -> Developer Tools
# Network -> Headers -> Request Payload

defaults_cmd=  {#'binary_frac': 0.3,
                #'binary_kind': 1,
                #'binary_mrinf': 0.7,
                #'binary_mrsup': 1,
                'cmd_version': 2.7,
                'dust_source': 'nodust',
                'dust_sourceC': 'nodustC',
                'dust_sourceM': 'nodustM',
                'eta_reimers': 0.2,
                #'extinction_av': 0,
                #'icm_lim': 4,
                'imf_file': 'tab_imf/imf_chabrier_lognormal.dat',
                'isoc_age': 1e9,
                'isoc_age0': 12.7e9,
                'isoc_dlage': 0.05,
                'isoc_dz': 0.0001,
                'isoc_kind': 'parsec_CAF09_v1.2S',
                'isoc_lage0': 6.602,   #Minimum allowed age                 
                'isoc_lage1': 10.1303, #Maximum allowed age                 
                'isoc_val': 0,               
                'isoc_z0': 0.0001,     #Minimum allowed metallicity         
                'isoc_z1': 0.03,       #Maximum allowed metallicity   
                'isoc_zeta': 0.0002,
                'isoc_zeta0': 0.0002,
                'kind_cspecmag': 'aringer09',
                'kind_dust': 0,
                'kind_interp': 1,
                'kind_mag': 2,
                'kind_postagb': -1,
                'kind_pulsecycle': 0,
                #'kind_tpagb': 0,
                #'lf_deltamag': 0.2,
                #'lf_maginf': 20,
                #'lf_magsup': -20,
                #'mag_lim': 26,
                #'mag_res': 0.1,
                'output_evstage': 1,
                'output_gzip': 0,
                'output_kind': 0,
                'photsys_file': 'tab_mag_odfnew/tab_mag_decam.dat',
                #'photsys_version': 'yang',
                'submit_form': 'Submit'}

defaults_27 = dict(defaults_cmd,cmd_version='2.7')
defaults_28 = dict(defaults_cmd,cmd_version='2.8')
defaults_29 = dict(defaults_cmd,cmd_version='2.9')
defaults_30 = dict(defaults_cmd,cmd_version='3.0')

# survey system
odict([
        ('des',dict(photosys_file='tab_mag_odfnew/tab_mag_decam.dat')),
        ('sdss',dict(photosys_file='')),
        ('pan-starrs',dict(photosys_file='')),
        ])


class Padova(object):
    defaults = dict(defaults_27)

    params2filename = PadovaIsochrone.params2filename
    filename2params = PadovaIsochrone.filename2params

    def __init__(self,survey='des',**kwargs):
        self.survey=survey

    def create_grid(self,abins,zbins):
        arange = np.linspace(abins[0],abins[1],abins[2]+1)
        zrange = np.logspace(np.log10(zbins[0]),np.log10(zbins[1]),zbins[2]+1)
        aa,zz = np.meshgrid(arange,zrange)
        return aa.flatten(),zz.flatten()

    def run(self,grid,outdir=None,force=False):
        aa,zz = grid
        for a,z in zip(aa,zz):
            try: 
                self.download(a,z,outdir,force)
            except RuntimeError, msg:
                logger.warning(msg)

    def download(self,age,metallicity,outdir=None,force=False):
        """
        Check valid parameter range and download isochrones from:
        http://stev.oapd.inaf.it/cgi-bin/cmd
        """
        epsilon = 1e-4
        lage = np.log10(age*1e9)
        lage_min,lage_max = self.defaults['isoc_lage0'],self.defaults['isoc_lage1']
        if not (lage_min-epsilon < lage <lage_max+epsilon):
            msg = 'Age outside of valid range: %g [%g < log(age) < %g]'%(lage,lage_min,lage_max)
            raise RuntimeError(msg)

        z_min,z_max = self.defaults['isoc_z0'],self.defaults['isoc_z1']
        if not (z_min <= metallicity <= z_max):
            msg = 'Metallicity outside of valid range: %g [%g < z < %g]'%(metallicity,z_min,z_max)
            raise RuntimeError(msg)

        survey=self.survey.lower()
        if survey=='des':
            photsys_file='tab_mag_odfnew/tab_mag_decam.dat'
        elif survey=='sdss':
            photsys_file='tab_mag_odfnew/tab_mag_sloan.dat'
        elif survey=='ps1':
            photsys_file='tab_mag_odfnew/tab_mag_panstarrs1.dat'
        else:
            msg = 'Unrecognized survey: %s'%survey
            raise RuntimeError(msg)

        if outdir is None: outdir = './'
        mkdir(outdir)

        basename = self.params2filename(age,metallicity)
        outfile = os.path.join(outdir,basename)
            
        if os.path.exists(outfile) and not force:
            logger.warning("Found %s; skipping..."%(outfile))
            return

        logger.info("Downloading isochrone: %s (age=%.2fGyr, metallicity=%g)"%(basename,age,metallicity))

        d = dict(self.defaults)
        d['photsys_file'] = photsys_file
        d['isoc_age']     = age * 1e9
        d['isoc_zeta']    = metallicity

        server = 'http://stev.oapd.inaf.it'
        url = server + '/cgi-bin/cmd_%s'%d['cmd_version']
        logger.info("Accessing %s..."%url)

        q = urlencode(d)
        logger.debug(url+'?'+q)
        c = urlopen(url, q).read()
        aa = re.compile('output\d+')
        fname = aa.findall(c)
        if len(fname) > 0:
            out = '{0}/~lgirardi/tmp/{1}.dat'.format(server, fname[0])
            cmd = 'wget %s -O %s'%(out,outfile)
            logger.debug(cmd)
            stdout = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
            logger.debug(stdout)
        else:
            raise RuntimeError('Server response is incorrect')

class OldPadova(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'gi10a'

    params2filename = OldPadovaIsochrone.params2filename
    filename2params = OldPadovaIsochrone.filename2params

class Girardi2002(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'gi2000'

class Marigo2008(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'ma08'

class Girardi2010a(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'gi10a'

class Girardi2010b(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'gi10b'

class Bressan2012(Padova):
    defaults = dict(defaults_27)
    defaults['isoc_kind'] = 'parsec_CAF09_v1.2S'

class Marigo2017(Padova):
    defaults = dict(defaults_30)
    defaults['isoc_kind'] = 'parsec_CAF09_v1.2S_NOV13'

def factory(name, **kwargs):
    from ugali.utils.factory import factory
    return factory(name, module=__name__, **kwargs)

if __name__ == "__main__":
    import ugali.utils.parser
    description = "Download isochrones"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_verbose()
    parser.add_force()
    parser.add_argument('-a','--age',default=None,type=float)
    parser.add_argument('-z','--metallicity',default=None,type=float)
    parser.add_argument('-k','--kind',default='Bressan2012')
    parser.add_argument('-s','--survey',default='des')
    parser.add_argument('-o','--outdir',default=None)
    parser.add_argument('-n','--njobs',default=10,type=int)
    args = parser.parse_args()

    # Defaults
    abins = np.linspace(1, 13.5, 126)
    zbins = np.linspace(0.0001,0.0010,91)
    abins = [args.age] if args.age else abins
    zbins = [args.metallicity] if args.metallicity else zbins
    grid = [g.flatten() for g in np.meshgrid(abins,zbins)]
    logger.info("Ages:\n  %s"%np.unique(grid[0]))
    logger.info("Metallicities:\n  %s"%np.unique(grid[1]))

    if args.outdir is None: args.outdir = args.kind.lower()
    logger.info("Creating output directory: %s"%args.outdir)

    p = factory(args.kind,survey=args.survey)

    def run(args):
        try:  
            p.download(*args)
        except RuntimeError as e: 
            logger.warn(str(e))

    arguments = [(a,z,args.outdir,args.force) for a,z in zip(*grid)]
    if args.njobs > 0:
        pool = Pool(processes=args.njobs, maxtasksperchild=100)
        results = pool.map(run,arguments)
    else:
        results = map(run,arguments)
        
#####################################################################3

    #from ugali.utils.config import Config
    #config = Config(args.config)
    #survey = config['data']['survey']

    #outdir = '/u/ki/kadrlica/des/isochrones/v1'
    #outdir = '/u/ki/kadrlica/sdss/isochrones/v2'
    #outdir = '/u/ki/kadrlica/des/isochrones/v2'
    #outdir = '/u/ki/kadrlica/des/isochrones/v3'
    #outdir = '/u/ki/kadrlica/des/isochrones/v4'
    #outdir = '/u/ki/kadrlica/des/isochrones/v5'
    #outdir = '/u/ki/kadrlica/des/isochrones/v6'
    #outdir = './iso'
    #outdir = args.outdir
    # Binning from config
    #abins = config['binning']['age']
    #zbins = config['binning']['z']
    #abins = np.arange(1,13.6,0.1)
    #zbins = np.arange(1e-4,1e-3,1e-5)
    #abins = np.arange(10,14,1)
    #zbins = np.arange(1e-4,2e-4,1e-4)
    #abins = np.arange(10,10.1,0.1)
    #zbins = np.arange(1e-4,1.1e-4,1e-5)
    #zbins = np.logspace(np.log10(0.001), np.log10(0.01), 50)
    #zbins = np.arange(0.00010,0.00100,0.00001)

    #grid = p.create_grid(abins,zbins)
    #grid = OldPadovaIsochrone.create_grid()
    #grid = np.meshgrid((10**np.arange(9.9,10.15,0.05))/1e9,np.array([0.12,0.15,0.19,0.24,0.30,0.38,0.48,0.6])*1e-3)
    #cut = (grid[0] > 0.5)
    #grid = (grid[0][cut],grid[1][cut])
    #grid = np.meshgrid(np.arange(1,13.5),np.arange(1e-4,1e-3,5e-5)
    #p.run(grid=grid,outdir=args.outdir,force=args.force)
    #parser.add_config()
    #parser.add_debug()
