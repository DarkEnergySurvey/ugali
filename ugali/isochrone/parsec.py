"""
Module for wrapping PARSEC isochrones.
"""
import os
import sys
import glob
import copy
from collections import OrderedDict as odict

# For downloading isochrones...
from urllib import urlencode
from urllib2 import urlopen
import subprocess
import re

import numpy as np
import scipy.spatial

from ugali.utils.logger import logger
from ugali.isochrone.model import Isochrone
from ugali.isochrone.model import get_iso_dir

# survey system
photsys_dict = odict([
        ('des' ,'tab_mag_odfnew/tab_mag_decam.dat'),
        ('sdss','tab_mag_odfnew/tab_mag_sloan.dat'),
        ('ps1' ,'tab_mag_odfnew/tab_mag_panstarrs1.dat'),
        ('acs_wfc' ,'tab_mag_odfnew/tab_mag_acs_wfc.dat'),
])

photname_dict = odict([
        ('des' ,'DECAM'),
        ('sdss','SDSS'),
        ('ps1' ,'Pan-STARRS1'),
        ('acs_wfc','HST/ACS'),
])

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
                'photsys_file': photsys_dict['des'],
                #'photsys_version': 'yang',
                'submit_form': 'Submit'}

defaults_27 = dict(defaults_cmd,cmd_version='2.7')
defaults_28 = dict(defaults_cmd,cmd_version='2.8')
defaults_29 = dict(defaults_cmd,cmd_version='2.9')
defaults_30 = dict(defaults_cmd,cmd_version='3.0')

class ParsecIsochrone(Isochrone):
    """ Base class for PARSEC-style isochrones. """

    download_defaults = copy.deepcopy(defaults_27)
    download_defaults['isoc_kind'] = 'parsec_CAF09_v1.2S'

    abins = np.arange(1.0, 13.5 + 0.1, 0.1)
    zbins = np.arange(1e-4,1e-3 + 1e-5,1e-5)

    @classmethod
    def z2feh(cls, z):
        # Taken from Table 3 and Section 3 of Bressan et al. 2012
        # Confirmed in Section 2.1 of Marigo et al. 2017
        Z_init  = z                # Initial metal abundance
        Y_p     = 0.2485           # Primordial He abundance (Komatsu 2011)
        c       = 1.78             # He enrichment ratio 

        Y_init = Y_p + c * Z_init 
        X_init = 1 - Y_init - Z_init

        Z_solar = 0.01524          # Solar metal abundance
        Y_solar = 0.2485           # Solar He abundance (Caffau 2011)
        X_solar = 1 - Y_solar - Z_solar

        return np.log10( Z_init/Z_solar * X_solar/X_init)
        
    @classmethod
    def feh2z(cls, feh):
        # Taken from Table 3 and Section 3 of Bressan et al. 2012
        # Confirmed in Section 2.1 of Marigo et al. 2017
        Y_p     = 0.2485           # Primordial He abundance
        c       = 1.78             # He enrichment ratio

        Z_solar = 0.01524          # Solar metal abundance
        Y_solar = 0.2485           # Solar He abundance
        X_solar = 1 - Y_solar - Z_solar

        return (1 - Y_p)/( (1 + c) + X_solar/Z_solar * 10**(-feh))

    def query_server(self,outfile,age,metallicity):
        """ Server query for the isochrone file. """

        params = copy.deepcopy(self.download_defaults)

        epsilon = 1e-4
        lage = np.log10(age*1e9)
        lage_min,lage_max = params['isoc_lage0'],params['isoc_lage1']
        if not (lage_min-epsilon < lage <lage_max+epsilon):
            msg = 'Age outside of valid range: %g [%g < log(age) < %g]'%(lage,lage_min,lage_max)
            raise RuntimeError(msg)

        z_min,z_max = params['isoc_z0'],params['isoc_z1']
        if not (z_min <= metallicity <= z_max):
            msg = 'Metallicity outside of valid range: %g [%g < z < %g]'%(metallicity,z_min,z_max)
            raise RuntimeError(msg)
        
        params['photsys_file'] = photsys_dict[self.survey]
        params['isoc_age']     = age * 1e9
        params['isoc_zeta']    = metallicity

        server = 'http://stev.oapd.inaf.it'
        url = server + '/cgi-bin/cmd_%s'%params['cmd_version']
        logger.debug("Accessing %s..."%url)

        q = urlencode(params)
        logger.debug(url+'?'+q)
        c = urlopen(url, q).read()
        aa = re.compile('output\d+')
        fname = aa.findall(c)
        
        if len(fname) == 0:
            msg = "Output filename not found"
            raise RuntimeError(msg)

        out = '{0}/~lgirardi/tmp/{1}.dat'.format(server, fname[0])
        cmd = 'wget %s -O %s'%(out,outfile)
        logger.debug(cmd)
        stdout = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
        logger.debug(stdout)

        return outfile

    @classmethod
    def verify(cls, filename, survey, age, metallicity):
        age = age*1e9
        nlines=15
        with open(filename,'r') as f:
            lines = [f.readline() for i in range(nlines)]
            if len(lines) < nlines:
                msg = "Incorrect file size"
                raise Exception(msg)

            for i,l in enumerate(lines):
                if l.startswith('# Photometric system:'): break
            else:
                msg = "Incorrect file header"
                raise Exception(msg)

            try:
                s = lines[i].split()[3]
                assert photname_dict[survey] == s
            except:
                msg = "Incorrect survey:\n"+lines[i]
                raise Exception(msg)

            try:
                z = lines[-1].split()[0]
                assert np.allclose(metallicity,float(z),atol=1e-5)
            except:
                msg = "Metallicity does not match:\n"+lines[-1]
                raise Exception(msg)

            try:
                a = lines[-1].split()[1]
                # Need to deal with age or log-age
                assert (np.allclose(age,float(a),atol=1e-2) or
                        np.allclose(np.log10(age),float(a),atol=1e-2))
            except:
                msg = "Age does not match:\n"+lines[-1]
                raise Exception(msg)


class Bressan2012(ParsecIsochrone):
    _dirname =  os.path.join(get_iso_dir(),'{survey}','bressan2012')

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage',4,'Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    download_defaults = copy.deepcopy(defaults_27)
    download_defaults['isoc_kind'] = 'parsec_CAF09_v1.2S'

    columns = dict(
        des = odict([
                (3, ('mass_init',float)),
                (4, ('mass_act',float)),
                (5, ('log_lum',float)),
                (10, ('g',float)),
                (11, ('r',float)),
                (12,('i',float)),
                (13,('z',float)),
                (14,('Y',float)),
                (16,('stage',int)),
                ]),
        sdss = odict([
                (3, ('mass_init',float)),
                (4, ('mass_act',float)),
                (5, ('log_lum',float)),
                (9, ('u',float)),
                (10,('g',float)),
                (11,('r',float)),
                (12,('i',float)),
                (13,('z',float)),
                (15,('stage',int)),
                ]),
        ps1 = odict([
                (3, ('mass_init',float)),
                (4, ('mass_act',float)),
                (5, ('log_lum',float)),
                (9, ('g',float)),
                (10,('r',float)),
                (11,('i',float)),
                (12,('z',float)),
                (13,('y',float)),
                (16,('stage',int)),
                ]),
        )

    def _parse(self,filename):
        """Reads an isochrone file in the Padova (Bressan et al. 2012)
        format. Creates arrays with the initial stellar mass and
        corresponding magnitudes for each step along the isochrone.
        """
        #http://stev.oapd.inaf.it/cgi-bin/cmd_2.7
        try:
            columns = self.columns[self.survey.lower()]
        except KeyError as e:
            logger.warning('Unrecognized survey: %s'%(survey))
            raise(e)

        # delimiter='\t' is used to be compatible with OldPadova...
        # ADW: This should be updated, but be careful of column numbering
        kwargs = dict(delimiter='\t',usecols=columns.keys(),
                      dtype=columns.values())
        self.data = np.genfromtxt(filename,**kwargs)

        self.mass_init = self.data['mass_init']
        self.mass_act  = self.data['mass_act']
        self.luminosity = 10**self.data['log_lum']
        self.mag_1 = self.data[self.band_1]
        self.mag_2 = self.data[self.band_2]
        self.stage = self.data['stage']

        self.mass_init_upper_bound = np.max(self.mass_init)
        self.index = len(self.mass_init)

        self.mag = self.mag_1 if self.band_1_detection else self.mag_2
        self.color = self.mag_1 - self.mag_2

class Marigo2017(ParsecIsochrone):
    #http://stev.oapd.inaf.it/cgi-bin/cmd_30
    #_dirname = '/u/ki/kadrlica/des/isochrones/v4/'
    _dirname =  os.path.join(get_iso_dir(),'{survey}','marigo2017')

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage',4,'Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    download_defaults = dict(defaults_30)
    download_defaults['isoc_kind'] = 'parsec_CAF09_v1.2S_NOV13'

    columns = dict(
        des = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (4, ('log_lum',float)),
                (7, ('stage',int)),
                (23,('u',float)),
                (24,('g',float)),
                (25,('r',float)),
                (26,('i',float)),
                (27,('z',float)),
                (28,('Y',float)),
                ]),
        sdss = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (4, ('log_lum',float)),
                (7, ('stage',int)),
                (23,('u',float)),
                (24,('g',float)),
                (25,('r',float)),
                (26,('i',float)),
                (27,('z',float)),
                ]),
        ps1 = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (4, ('log_lum',float)),
                (7, ('stage',int)),
                (23,('g',float)),
                (24,('r',float)),
                (25,('i',float)),
                (26,('z',float)),
                (27,('y',float)),
                (28,('w',float)),
                ]),
        )
    
    def _parse(self,filename):
        """Reads an isochrone file in the Padova (Marigo et al. 2017)
        format. Creates arrays with the initial stellar mass and
        corresponding magnitudes for each step along the isochrone.
        """
        try:
            columns = self.columns[self.survey.lower()]
        except KeyError as e:
            logger.warning('Unrecognized survey: %s'%(survey))
            raise(e)

        kwargs = dict(usecols=columns.keys(),dtype=columns.values())
        self.data = np.genfromtxt(filename,**kwargs)

        self.mass_init = self.data['mass_init']
        self.mass_act  = self.data['mass_act']
        self.luminosity = 10**self.data['log_lum']
        self.mag_1 = self.data[self.band_1]
        self.mag_2 = self.data[self.band_2]
        self.stage = self.data['stage']

        self.mass_init_upper_bound = np.max(self.mass_init)
        self.index = len(self.mass_init)

        self.mag = self.mag_1 if self.band_1_detection else self.mag_2
        self.color = self.mag_1 - self.mag_2
