#!/usr/bin/env python
import os
import sys
import glob
import copy
from collections import OrderedDict as odict

import re
from urllib import urlencode
from urllib2 import urlopen
#import requests
import tempfile
import subprocess
from collections import OrderedDict as odict

import numpy as np

from ugali.utils.logger import logger
from ugali.isochrone.model import Isochrone
from ugali.isochrone.model import get_iso_dir

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

dict_clr = {
    'acs_wfc':4,
    'des' : 14,
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


class Dotter2008(Isochrone):
    """
    KCB: currently inheriting from PadovaIsochrone because there are 
    several useful functions where we would basically be copying code.
    """
    _dirname =  os.path.join(get_iso_dir(),'{survey}','dotter2008')
    #_zsolar = 0.0163 
    _zsolar = 0.0180 # Grevesse & Sauval, 1998

    # KCB: What to do about horizontal branch?
    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage','BHeb','Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    abins = np.arange(1., 13.5 + 0.1, 0.1)
    zbins = np.arange(7e-5,1e-3 + 1e-5,1e-5)

    download_defaults = copy.deepcopy(dartmouth_defaults)

    columns = dict(
            des = odict([
                (1, ('mass',float)),
                (4, ('log_lum',float)),
                (5, ('u',float)),
                (6, ('g',float)),
                (7, ('r',float)),
                (8, ('i',float)),
                (9, ('z',float)),
                ]),
            sdss = odict([
                (1, ('mass',float)),
                (4, ('log_lum',float)),
                (5, ('u',float)),
                (6, ('g',float)),
                (7, ('r',float)),
                (8, ('i',float)),
                (9, ('z',float)),
                ]),
            ps1 = odict([
                (1, ('mass',float)),
                (4, ('log_lum',float)),
                (6, ('g',float)),
                (7, ('r',float)),
                (8, ('i',float)),
                (9, ('z',float)),
                (10, ('y',float)),
                ]),
            )

    def _parse(self,filename):
        """
        Reads an isochrone in the Dotter format and determines the 
        age (log10 yrs and Gyr), metallicity (Z and [Fe/H]), and 
        creates arrays with the initial stellar mass and 
        corresponding magnitudes for each step along the isochrone.
        http://stellar.dartmouth.edu/models/isolf_new.html
        """
        try:
            columns = self.columns[self.survey.lower()]
        except KeyError, e:
            logger.warning('Unrecognized survey: %s'%(survey))
            raise(e)

        kwargs = dict(comments='#',usecols=columns.keys(),dtype=columns.values())
        self.data = np.genfromtxt(filename,**kwargs)

        # KCB: Not sure whether the mass in Dotter isochrone output
        # files is initial mass or current mass
        self.mass_init = self.data['mass']
        self.mass_act  = self.data['mass']
        self.luminosity = 10**self.data['log_lum']
        self.mag_1 = self.data[self.band_1]
        self.mag_2 = self.data[self.band_2]
        self.stage = np.tile('Main', len(self.data))
        
        # KCB: No post-AGB isochrone data points, right?
        self.mass_init_upper_bound = np.max(self.mass_init)

        self.mag = self.mag_1 if self.band_1_detection else self.mag_2
        self.color = self.mag_1 - self.mag_2

    @classmethod
    def z2feh(cls, z):
        # Section 3 of Dotter et al. 2008
        # Section 2 of Dotter et al. 2007 (0706.0847)
        Z_init  = z                # Initial metal abundance
        Y_p     = 0.245            # Primordial He abundance (WMAP, 2003)
        c       = 1.54             # He enrichment ratio 

        Y_init = Y_p + c * Z_init 
        X_init = 1 - Y_init - Z_init

        # This is not well defined...
        #Z_solar/X_solar = 0.0229  # Solar metal fraction (Grevesse 1998)
        ZX_solar = 0.0229
        return np.log10( Z_init/X_init * 1/ZX_solar)

    @classmethod
    def feh2z(cls, feh):
        # Section 3 of Dotter et al. 2008
        Y_p     = 0.245            # Primordial He abundance (WMAP, 2003)
        c       = 1.54             # He enrichment ratio 

        # This is not well defined...
        #Z_solar/X_solar = 0.0229  # Solar metal fraction (Grevesse 1998)
        ZX_solar = 0.0229
        return (1 - Y_p)/( (1 + c) + (1/ZX_solar) * 10**(-feh))


    def query_server(self, outfile, age, metallicity):
        z = metallicity
        feh = self.z2feh(z)
        
        params = copy.deepcopy(self.download_defaults)
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

