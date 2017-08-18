#!/usr/bin/env python
"""
Script to automate the process of generating an isochrone library
using the Dotter isochrones.

Download isochrones from:
http://stellar.dartmouth.edu/models/isolf_new.html

New MESA isochrones come from:
http://waps.cfa.harvard.edu/MIST/interp_isos.html
"""

import os
import re
import urllib2
from urllib import urlencode
from urllib2 import urlopen
import requests
import sys
import copy
import tempfile
import subprocess
import shutil
from multiprocessing import Pool
from collections import OrderedDict as odict

import numpy
import numpy as np

import ugali.utils.logger
import ugali.utils.shell
from ugali.utils.logger import logger
from ugali.utils.shell import mkdir
import ugali.analysis.isochrone as iso
from ugali.analysis.isochrone import DotterIsochrone
from ugali.preprocess.padova import Download

"""
See Vargas et al. 2013 for the distribution of alpha elements in
dSphs: http://adsabs.harvard.edu/abs/2013ApJ...767..134V

Josh Simon remarks: For stars at [Fe/H] > -2, [a/Fe] tends to be
around zero. [Note, though, that this paper does not attempt to do
any membership classification, it just accepts the lists from
Simon & Geha 2007.  I suspect now that we were not sufficiently
conservative on selecting members in those early days, and so some
of the relatively metal-rich stars may in fact be foreground Milky
Way stars.]  More metal-poor stars tend to average more like
[a/Fe] = 0.4-0.5.  Fig. 5 of Frebel et al. (2014) shows similar
plots for individual elements from high-resolution spectra.  Given
these data, plus the empirical fact that the mean metallicities of
the ultra-faint dwarfs are almost universally [Fe/H] < -2, I guess
I would say [a/Fe] = 0.3 is probably the best compromise.

From ADW: In order to makea comparison with other isochrones (which
don't provide a knob for [a/Fe], I suggest that we stick to [a/Fe] = 0
for the Dotter2008 isochrones.
"""

dict_clr = {'des' : 14,
            'sdss': 11,
            'ps1' : 12,
            }

dict_hel = {'Y=0.245+1.5*Z' : 1,
            'Y=0.33'        : 2,
            'Y=0.40'        : 3}

dict_afe = {'-0.2'            : 1,
            '0 (scaled-solar)': 2,
            '+0.2'            : 3,
            '+0.4'            : 4,
            '+0.6'            : 5,
            '+0.8'            : 6}

# survey system
dict_output = odict([
        ('des','DECam'),
        ('sdss','SDSSugriz'),
        ('ps1','PanSTARRS'),
])

# Dartmouth Isochrones
# http://stellar.dartmouth.edu/models/isolf_new.php
dartmouth_defaults = {
    'int':'2', # interpolation: cubic=1, linear=2
    'out':'1', # outpur: iso=1, iso+LF=2
    'age':10, # age [Gyr]
    'feh':-3.0, # feh
    'hel': dict_hel['Y=0.245+1.5*Z'], # initial helium abundance
    'afe': dict_afe['0 (scaled-solar)'], # alpha enhancement
    'clr': dict_clr['des'], # photometric system
    'flt':'',
    'bin':'',
    'imf':1,
    'pls':'',
    'lnm':'',
    'lns':'', 
    }


# MESA Isochrones from
# http://waps.cfa.harvard.edu/MIST/iso_form.php
mesa_defaults = {
        'version':'1.0',
        'v_div_vcrit':'vvcrit0.4',
        'age_scale':'linear',
        'age_type':'single',
        'age_value':10e9, # yr if scale='linear'; log10(yr) if scale='log10'
        'age_range_low':'',
        'age_range_high':'',
        'age_range_delta':'',
        'age_list':'',
        'FeH_value':-3.0,
        'theory_output':'basic',
        'output_option':'photometry',
        'output':'DECam',
        'Av_value':0,
}

mesa_defaults_10 = dict(mesa_defaults,version='1.0')

class Dotter(Download):
    defaults = copy.deepcopy(dartmouth_defaults)

    params2filename = iso.Dotter2008.params2filename
    filename2params = iso.Dotter2008.filename2params

    abins = np.arange(1., 13.5 + 0.1, 0.1)
    zbins = np.arange(7e-5,1e-3 + 1e-5,1e-5)

    def download(self,age,metallicity,outdir=None,force=False):

        tmpfile = tempfile.NamedTemporaryFile().name

        z = metallicity
        logger.info('Downloading isochrone: %s (age=%.1fGyr, metallicity=%.5f)'%(self.__class__.__name__, age, z))
        feh = iso.Dotter2008.z2feh(z)

        params = dict(self.defaults)
        params['age']=age
        params['feh']=feh
        params['clr']=dict_clr[self.survey]

        url = 'http://stellar.dartmouth.edu/models/isolf_new.php'
        query = url + '?' + urlencode(params)
        logger.debug(query)
        response = urlopen(query)
        page_source = response.read()
        isochrone_id = page_source.split('tmp/tmp')[-1].split('.iso')[0]

        infile = 'http://stellar.dartmouth.edu/models/tmp/tmp%s.iso'%(isochrone_id)
        command = 'wget -q %s -O %s'%(infile, tmpfile)
        subprocess.call(command,shell=True)
        
        if outdir is None: outdir = './'

        ## Rename the output file based on Z effective ([a/Fe] corrected)
        #tmp = open(tmpfile,'r')
        #lines = [tmp.readline() for i in range(4)]
        #z_eff = float(lines[3].split()[4])
        #basename = self.params2filename(age,z_eff)

        basename = self.params2filename(age,z)
        outfile = os.path.join(outdir,basename)

        if os.path.exists(outfile) and not force:
            logger.warning("Found %s; skipping..."%(outfile))
            return

        logger.info("Writing %s..."%outfile)
        mkdir(outdir)
        shutil.move(tmpfile,outfile)

class Dotter2008(Dotter):
    """ Dartmouth isochrones from Dotter et al. 2008:
    http://stellar.dartmouth.edu/models/isolf_new.html
    """

class Dotter2016(Download):
    """ MESA isochrones from Dotter 2016:
    http://waps.cfa.harvard.edu/MIST/iso_form.php
    """
    defaults = copy.deepcopy(mesa_defaults_10)

    params2filename = DotterIsochrone.params2filename
    filename2params = DotterIsochrone.filename2params

    abins = np.arange(1., 13.5 + 0.1, 0.1)
    zbins = np.arange(1e-5,1e-3 + 1e-5,1e-5)

    def download(self,age,metallicity,outdir=None,force=False):
        z = metallicity

        if outdir is None: outdir = './'
        basename = self.params2filename(age,z)
        outfile = os.path.join(outdir,basename)

        if os.path.exists(outfile) and not force:
            logger.warning("Found %s; skipping..."%(outfile))
            return
        mkdir(outdir)

        logger.info('Downloading isochrone: %s (age=%.1fGyr, metallicity=%.5f)'%(self.__class__.__name__, age, z))

        feh = iso.Dotter2016.z2feh(z)
        params = dict(self.defaults)
        params['output'] = dict_output[self.survey]
        params['FeH_value'] = feh
        params['age_value'] = age * 1e9
        if params['age_scale'] == 'log10':
            params['age_value'] = np.log10(params['age_value'])

        server = 'http://waps.cfa.harvard.edu/MIST'
        url = server + '/iso_form.php'
        logger.info("Accessing %s..."%url)
        response = requests.post(url,data=params)

        fname = os.path.basename(response.text.split('"')[1])
        tmpdir = os.path.dirname(tempfile.NamedTemporaryFile().name)
        tmpfile = os.path.join(tmpdir,fname)

        if len(fname) > 0:
            out = '{0}/tmp/{1}'.format(server, fname)
            cmd = 'wget %s -P %s'%(out,tmpdir)
            logger.debug(cmd)
            stdout = subprocess.check_output(cmd,shell=True,
                                             stderr=subprocess.STDOUT)
            logger.debug(stdout)

        else:
            raise RuntimeError('Server response is incorrect')

        cmd = 'unzip %s -d %s'%(tmpfile,tmpdir)
        logger.debug(cmd)
        stdout = subprocess.check_output(cmd,shell=True,
                                         stderr=subprocess.STDOUT)
        logger.debug(stdout)

        logger.info("Creating %s..."%outfile)
        shutil.move(tmpfile.replace('.zip','.cmd'),outfile)
        os.remove(tmpfile)

def factory(name, **kwargs):
    from ugali.utils.factory import factory
    return factory(name, module=__name__, **kwargs)

if __name__ == "__main__":
    import ugali.utils.parser
    description = "Download Dotter isochrones"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_verbose()
    parser.add_force()
    parser.add_argument('-a','--age',default=None,type=float,action='append')
    parser.add_argument('-z','--metallicity',default=None,type=float,action='append')
    parser.add_argument('-k','--kind',default='Dotter2008')
    parser.add_argument('-s','--survey',default='des')
    parser.add_argument('-o','--outdir',default=None)
    parser.add_argument('-n','--njobs',default=1,type=int)
    args = parser.parse_args()

    if args.verbose:
        from httplib import HTTPConnection
        HTTPConnection.debuglevel = 1

    if args.outdir is None: 
        args.outdir = os.path.join(args.survey.lower(),args.kind.lower())

    p = factory(args.kind,survey=args.survey)

    abins = args.age if args.age else p.abins
    zbins = args.metallicity if args.metallicity else p.zbins
    grid = [g.flatten() for g in np.meshgrid(abins,zbins)]
    logger.info("Ages:\n  %s"%np.unique(grid[0]))
    logger.info("Metallicities:\n  %s"%np.unique(grid[1]))

    def run(args):
        try:  
            p.download(*args)
        except RuntimeError as e: 
            logger.warn(str(e))

    arguments = [(a,z,args.outdir,args.force) for a,z in zip(*grid)]
    if args.njobs > 1:
        msg = "Multiprocessing does not work for Dotter2008 download."
        raise Exception(msg)
    #elif args.njobs > 1:
    #    pool = Pool(processes=args.njobs, maxtasksperchild=100)
    #    results = pool.map(run,arguments)
    else:
        results = map(run,arguments)
