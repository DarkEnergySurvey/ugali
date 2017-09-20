#!/usr/bin/env python
"""
MESA Isochrones from:
http://waps.cfa.harvard.edu/MIST/iso_form.php
"""
import os
import sys
import glob
import copy
import tempfile
import subprocess
import shutil
from collections import OrderedDict as odict

import urllib, urllib2
#import requests
import numpy as np


from ugali.utils.logger import logger

from ugali.isochrone.parsec import Isochrone
from ugali.isochrone.model import get_iso_dir

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

class Dotter2016(Isochrone):
    """ MESA isochrones from Dotter 2016:
    http://waps.cfa.harvard.edu/MIST/interp_isos.html
    """
    _dirname =  os.path.join(get_iso_dir(),'{survey}','dotter2016')

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage',3,'Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    download_defaults = copy.deepcopy(mesa_defaults_10)

    abins = np.arange(1., 13.5+0.1, 0.1)
    zbins = np.arange(1e-5, 1e-3+1e-5, 1e-5)

    columns = dict(
            des = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (8, ('log_lum',float)),
                (9,('u',float)),
                (10,('g',float)),
                (11,('r',float)),
                (12,('i',float)),
                (13,('z',float)),
                (14,('Y',float)),
                (15,('stage',float))
                ]),
            sdss = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (6, ('log_lum',float)),
                (9, ('u',float)),
                (10,('g',float)),
                (11,('r',float)),
                (12,('i',float)),
                (13,('z',float)),
                (14,('stage',float))
                ]),
            ps1 = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (6, ('log_lum',float)),
                (9, ('g',float)),
                (10,('r',float)),
                (11,('i',float)),
                (12,('z',float)),
                (13,('y',float)),
                (16,('stage',float))
                ]),
            )

    def _parse(self,filename):
        """
        Reads an isochrone in the Dotter 2016 format and determines
        the age (Gyr), metallicity (Z), and creates arrays with the
        initial stellar mass and corresponding magnitudes for each
        step along the isochrone.
        """
        try:
            columns = self.columns[self.survey.lower()]
        except KeyError, e:
            logger.warning('Unrecognized survey: %s'%(survey))
            raise(e)

        kwargs = dict(comments='#',usecols=columns.keys(),dtype=columns.values())
        data = np.genfromtxt(filename,**kwargs)

        self.mass_init = data['mass_init']
        self.mass_act  = data['mass_act']
        self.luminosity = 10**data['log_lum']
        self.mag_1 = data[self.band_1]
        self.mag_2 = data[self.band_2]
        self.stage = data['stage']
        
        # Check where post-AGB isochrone data points begin
        self.mass_init_upper_bound = np.max(self.mass_init)
        self.index = np.nonzero(self.stage >= 4)[0][0]

        self.mag = self.mag_1 if self.band_1_detection else self.mag_2
        self.color = self.mag_1 - self.mag_2


    @classmethod
    def z2feh(cls, z):
        # Section 3.1 of Choi et al. 2016 (https://arxiv.org/abs/1604.08592)
        Z_init  = z                # Initial metal abundance
        Y_p     = 0.249            # Primordial He abundance (Planck 2015)
        c       = 1.5              # He enrichment ratio 

        Y_init = Y_p + c * Z_init 
        X_init = 1 - Y_init - Z_init

        Z_solar = 0.0142           # Solar metal abundance
        Y_solar = 0.2703           # Solar He abundance (Asplund 2009)
        X_solar = 1 - Y_solar - Z_solar

        return np.log10( Z_init/Z_solar * X_solar/X_init)

    @classmethod
    def feh2z(cls, feh):
        # Section 3.1 of Choi et al. 2016 (https://arxiv.org/abs/1604.08592)
        Y_p     = 0.249            # Primordial He abundance (Planck 2015)
        c       = 1.5              # He enrichment ratio 

        Z_solar = 0.0142           # Solar metal abundance
        Y_solar = 0.2703           # Solar He abundance (Asplund 2009)
        X_solar = 1 - Y_solar - Z_solar

        return (1 - Y_p)/( (1 + c) + (X_solar/Z_solar) * 10**(-feh))

    def query_server(self, outfile, age, metallicity):
        z = metallicity
        feh = self.z2feh(z)
        
        params = dict(self.download_defaults)
        params['output'] = dict_output[self.survey]
        params['FeH_value'] = feh
        params['age_value'] = age * 1e9
        if params['age_scale'] == 'log10':
            params['age_value'] = np.log10(params['age_value'])

        server = 'http://waps.cfa.harvard.edu/MIST'
        url = server + '/iso_form.php'
        logger.debug("Accessing %s..."%url)
        #response = requests.post(url,data=params)
        q = urllib.urlencode(params)
        request = urllib2.Request(url,data=q)
        response = urllib2.urlopen(request)
        try:
            #fname = os.path.basename(response.text.split('"')[1])
            fname = os.path.basename(response.read().split('"')[1])
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
