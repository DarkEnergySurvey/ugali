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
try:
    from urllib.parse import urlencode
    from urllib.request import urlopen
except ImportError:
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

From ADW: Other isochrone sets impose [a/Fe] = 0. For an accurate
comparison between isochrones, I suggest that we stick to [a/Fe] = 0
for the Dotter2008 isochrones as well.

ADW: The Dotter 2008 isochrones are interpolated from a relative
sparse grid of metallicities [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]. This
can lead to rather large differences between the input and output Z ->
[Fe/H] conversions. We were initially taking the output Zeff value,
but we now use the input Z value for internal consistency.
"""


###########################################################
# Dartmouth Isochrones
# http://stellar.dartmouth.edu/models/isolf_new.php

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

dartmouth_defaults = {
    'int':'1', # interpolation: cubic=1, linear=2 (ADW: cubic more accurate)
    'out':'1', # outpur: iso=1, iso+LF=2
    'age':10, # age [Gyr]
    'feh':-2.0, # feh [-2.5 to 0.0]
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

###########################################################
# MESA Isochrones
# http://waps.cfa.harvard.edu/MIST/iso_form.php

# survey system
dict_output = odict([
        ('des','DECam'),
        ('sdss','SDSSugriz'),
        ('ps1','PanSTARRS'),
])

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

class Dotter2008(Download):
    """ Dartmouth isochrones from Dotter et al. 2008:
    http://stellar.dartmouth.edu/models/isolf_new.html
    """
    defaults = copy.deepcopy(dartmouth_defaults)
    isochrone=iso.Dotter2008

    abins = np.arange(1., 13.5 + 0.1, 0.1)
    zbins = np.arange(7e-5,1e-3 + 1e-5,1e-5)

    def query_server(self, outfile, age, metallicity):
        z = metallicity
        feh = self.isochrone.z2feh(z)
        
        params = dict(self.defaults)
        params['age']=age
        params['feh']='%.6f'%feh
        params['clr']=dict_clr[self.survey]

        url = 'http://stellar.dartmouth.edu/models/isolf_new.php'
        query = url + '?' + urlencode(params)
        logger.debug(query)
        response = urlopen(query)
        page_source = response.read()
        try:
            file_id = int(page_source.split('tmp/tmp')[-1].split('.iso')[0])
        except Exception as e:
            logger.debug(str(e))
            msg = 'Output filename not found'
            raise RuntimeError(msg)

        infile = 'http://stellar.dartmouth.edu/models/tmp/tmp%s.iso'%(file_id)
        command = 'wget -q %s -O %s'%(infile, outfile)
        subprocess.call(command,shell=True)        

        ## ADW: Old code to rename the output file based on Zeff ([a/Fe] corrected)
        #tmpfile = tempfile.NamedTemporaryFile().name
        #tmp = open(tmpfile,'r')
        #lines = [tmp.readline() for i in range(4)]
        #z_eff = float(lines[3].split()[4])
        #basename = self.params2filename(age,z_eff)
     
        #logger.info("Writing %s..."%outfile)
        #mkdir(outdir)
        #shutil.move(tmpfile,outfile)

    @classmethod
    def verify(cls, filename, survey, age, metallicity):
        nlines=8
        with open(filename,'r') as f:
            lines = [f.readline() for i in range(nlines)]

            if len(lines) < nlines:
                msg = "Incorrect file size"
                raise Exception(msg)

            try:
                z = lines[3].split()[4]
                assert np.allclose(metallicity,float(z),atol=1e-3)
            except:
                msg = "Metallicity does not match:\n"+lines[3]
                raise Exception(msg)

            try:
                s = lines[5].split()[2]
                assert dict_output[survey][:4] in s
            except:
                msg = "Incorrect survey:\n"+lines[5]
                raise Exception(msg)

            try:
                a = lines[7].split('=')[1].strip().split()[0]
                assert np.allclose(age,float(a),atol=1e-5)
            except:
                msg = "Age does not match:\n"+lines[7]
                raise Exception(msg)

class Dotter2016(Download):
    """ MESA isochrones from Dotter 2016:
    http://waps.cfa.harvard.edu/MIST/iso_form.php
    """
    defaults = copy.deepcopy(mesa_defaults_10)
    isochrone = iso.Dotter2016

    abins = np.arange(1., 13.5 + 0.1, 0.1)
    zbins = np.arange(1e-5,1e-3 + 1e-5,1e-5)

    def query_server(self, outfile, age, metallicity):
        z = metallicity
        feh = self.isochrone.z2feh(z)
        
        params = dict(self.defaults)
        params['output'] = dict_output[self.survey]
        params['FeH_value'] = feh
        params['age_value'] = age * 1e9
        if params['age_scale'] == 'log10':
            params['age_value'] = np.log10(params['age_value'])

        server = 'http://waps.cfa.harvard.edu/MIST'
        url = server + '/iso_form.php'
        logger.debug("Accessing %s..."%url)
        response = requests.post(url,data=params)

        try:
            fname = os.path.basename(response.text.split('"')[1])
        except Exception as e:
            logger.debug(str(e))
            msg = 'Output filename not found'
            raise RuntimeError(msg)
            
        tmpdir = os.path.dirname(tempfile.NamedTemporaryFile().name)
        tmpfile = os.path.join(tmpdir,fname)

        out = '{0}/tmp/{1}'.format(server, fname)
        cmd = 'wget %s -P %s'%(out,tmpdir)
        logger.debug(cmd)
        stdout = subprocess.check_output(cmd,shell=True,
                                         stderr=subprocess.STDOUT)
        logger.debug(stdout)

        cmd = 'unzip %s -d %s'%(tmpfile,tmpdir)
        logger.debug(cmd)
        stdout = subprocess.check_output(cmd,shell=True,
                                         stderr=subprocess.STDOUT)
        logger.debug(stdout)

        logger.debug("Creating %s..."%outfile)
        shutil.move(tmpfile.replace('.zip','.cmd'),outfile)
        os.remove(tmpfile)

        return outfile

    @classmethod
    def verify(cls, filename, survey, age, metallicity):
        age = age*1e9
        nlines=14
        with open(filename,'r') as f:
            lines = [f.readline() for i in range(nlines)]
            if len(lines) < nlines:
                msg = "Incorrect file size"
                raise Exception(msg)
                
            try:
                s = lines[2].split()[-2]
                assert dict_output[survey][:4] in s
            except:
                msg = "Incorrect survey:\n"+lines[2]
                raise Exception(msg)

            try:
                z = lines[5].split()[2]
                assert np.allclose(metallicity,float(z),atol=1e-3)
            except:
                msg = "Metallicity does not match:\n"+lines[5]
                raise Exception(msg)

            try:
                a = lines[13].split()[1]
                assert np.allclose(age,float(a),atol=1e-5)
            except:
                msg = "Age does not match:\n"+lines[13]
                raise Exception(msg)

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
        try:
            from http.client import HTTPConnection
        except ImportError:
            from httplib import HTTPConnection
        HTTPConnection.debuglevel = 1

    if args.outdir is None: 
        args.outdir = os.path.join(args.survey.lower(),args.kind.lower())
    logger.info("Writing to output directory: %s"%args.outdir)

    p = factory(args.kind,survey=args.survey)

    abins = args.age if args.age else p.abins
    zbins = args.metallicity if args.metallicity else p.zbins
    grid = [g.flatten() for g in np.meshgrid(abins,zbins)]
    logger.info("Ages:\n  %s"%np.unique(grid[0]))
    logger.info("Metallicities:\n  %s"%np.unique(grid[1]))

    def run(args):
        try:  
            p.download(*args)
        except Exception as e:
            # power through any exceptions...
            logger.warn(str(e))

    arguments = [(a,z,args.outdir,args.force) for a,z in zip(*grid)]
    if args.njobs > 1:
        msg = "Multiprocessing does not work for %s download."%args.kind
        raise Exception(msg)
    #elif args.njobs > 1:
    #    pool = Pool(processes=args.njobs, maxtasksperchild=100)
    #    results = pool.map(run,arguments)
    else:
        results = list(map(run,arguments))
