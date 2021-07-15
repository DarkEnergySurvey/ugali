"""
Module for wrapping PARSEC isochrones.
http://stev.oapd.inaf.it
"""
import os
import sys
import glob
import copy
from collections import OrderedDict as odict

# For downloading isochrones...
try:
    from urllib.parse import urlencode
    from urllib.request import urlopen
    from urllib.error import URLError
except ImportError:
    from urllib import urlencode
    from urllib2 import urlopen, URLError

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
        ('lsst', 'tab_mag_odfnew/tab_mag_lsst.dat'),
])

photname_dict = odict([
        ('des' ,'DECAM'),
        ('sdss','SDSS'),
        ('ps1' ,'Pan-STARRS1'),
        ('acs_wfc','HST/ACS'),
        ('lsst', 'LSST'),
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

# Access prior to 3.1 seems to be gone
defaults_27 = dict(defaults_cmd,cmd_version=2.7)
defaults_28 = dict(defaults_cmd,cmd_version=2.8)
defaults_29 = dict(defaults_cmd,cmd_version=2.9)
defaults_30 = dict(defaults_cmd,cmd_version=3.0)

# This seems to maintain old ischrone format
defaults_31 = dict(defaults_cmd,cmd_version=3.1)

# New query and file format for 3.3...
defaults_33 = {'cmd_version': 3.3,
               'track_parsec': 'parsec_CAF09_v1.2S',
               'track_colibri': 'parsec_CAF09_v1.2S_S35',
               'track_postagb': 'no',
               'n_inTPC': 10,
               'eta_reimers': 0.2,
               'kind_interp': 1,
               'kind_postagb': -1,
               'photsys_file': photsys_dict['des'],
               'photsys_version': 'OBC',
               'dust_sourceM': 'dpmod60alox40',
               'dust_sourceC': 'AMCSIC15',
               'kind_mag': 2,
               'kind_dust': 0,
               #'extinction_av': 0.0,
               'extinction_coeff': 'constant',
               'extinction_curve': 'cardelli',
               'imf_file': 'tab_imf/imf_chabrier_lognormal.dat',
               'isoc_isagelog': 0,
               'isoc_agelow': 1.0e9,
               'isoc_ageupp': 1.0e10,
               'isoc_dage': 0.0,
               'isoc_lagelow': 6.6,
               'isoc_lageupp': 10.13,
               'isoc_dlage': 0.0,
               'isoc_ismetlog': 0,
               'isoc_zlow': 0.0152,
               'isoc_zupp': 0.03,
               'isoc_dz': 0.0,
               'isoc_metlow': -2,
               'isoc_metupp': 0.3,
               'isoc_dmet': 0.0,
               'output_kind': 0,
               'output_evstage': 1,
               #'lf_maginf': -15,
               #'lf_magsup': 20,
               #'lf_deltamag': 0.5,
               #'sim_mtot': 1.0e4,
               'submit_form': 'Submit',
               #'.cgifields': 'dust_sourceC',
               #'.cgifields': 'track_colibri',
               #'.cgifields': 'extinction_curve',
               #'.cgifields': 'output_kind',
               #'.cgifields': 'photsys_version',
               #'.cgifields': 'isoc_isagelog',
               #'.cgifields': 'track_parsec',
               #'.cgifields': 'extinction_coeff',
               #'.cgifields': 'track_postagb',
               #'.cgifields': 'output_gzip',
               #'.cgifields': 'isoc_ismetlog',
               #'.cgifields': 'dust_sourceM',
               }

class ParsecIsochrone(Isochrone):
    """ Base class for PARSEC-style isochrones. """

    download_url = "http://stev.oapd.inaf.it"
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
        """ Server query for the isochrone file.

        Parameters:
        -----------
        outfile     : name of output isochrone file
        age         : isochrone age
        metallicity : isochrone metallicity
        
        Returns:
        --------
        outfile     : name of output isochrone file
        """
        params = copy.deepcopy(self.download_defaults)

        epsilon = 1e-4
        lage = np.log10(age*1e9)
        
        lage_min = params.get('isoc_lage0',6.602)
        lage_max = params.get('isoc_lage1',10.1303)

        if not (lage_min-epsilon < lage <lage_max+epsilon):
            msg = 'Age outside of valid range: %g [%g < log(age) < %g]'%(lage,lage_min,lage_max)
            raise RuntimeError(msg)

        z_min = params.get('isoc_z0',0.0001)
        z_max = params.get('isoc_z1',0.03)
    
        if not (z_min <= metallicity <= z_max):
            msg = 'Metallicity outside of valid range: %g [%g < z < %g]'%(metallicity,z_min,z_max)
            raise RuntimeError(msg)
        
        params['photsys_file'] = photsys_dict[self.survey]
        if params['cmd_version'] < 3.3:
            params['isoc_age']    = age * 1e9
            params['isoc_zeta']   = metallicity
        else:
            params['isoc_agelow'] = age * 1e9
            params['isoc_zlow']   = metallicity
    
        server = self.download_url
        url = server + '/cgi-bin/cmd_%s'%params['cmd_version']
        # First check that the server is alive
        logger.debug("Accessing %s..."%url)
        urlopen(url,timeout=2)

        q = urlencode(params).encode('utf-8')
        logger.debug(url+'?'+q)
        c = str(urlopen(url, q).read())
        aa = re.compile('output\d+')
        fname = aa.findall(c)
        
        if len(fname) == 0:
            msg = "Output filename not found"
            raise RuntimeError(msg)

        out = '{0}/tmp/{1}.dat'.format(server, fname[0])
        
        cmd = 'wget --progress dot:binary %s -O %s'%(out,outfile)
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

    #download_defaults = copy.deepcopy(defaults_27)
    download_defaults = copy.deepcopy(defaults_31)
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
        lsst = odict([
                (3, ('mass_init',float)),
                (4, ('mass_act',float)),
                (5, ('log_lum',float)),
                (9, ('u',float)),
                (10,('g',float)),
                (11,('r',float)),
                (12,('i',float)),
                (13,('z',float)),
                (14,('Y',float)),
                (16,('stage',float))
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
            logger.warning('Unrecognized survey: %s'%(self.survey))
            raise(e)

        # delimiter='\t' is used to be compatible with OldPadova...
        # ADW: This should be updated, but be careful of column numbering
        kwargs = dict(delimiter='\t',usecols=list(columns.keys()),
                      dtype=list(columns.values()))
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
    #http://stev.oapd.inaf.it/cgi-bin/cmd_31
    #_dirname = '/u/ki/kadrlica/des/isochrones/v4/'
    _dirname =  os.path.join(get_iso_dir(),'{survey}','marigo2017')

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage',4,'Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    download_defaults = copy.deepcopy(defaults_31)
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
        lsst = odict([
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
        )

    def _parse(self,filename):
        """Reads an isochrone file in the Padova (Marigo et al. 2017)
        format. Creates arrays with the initial stellar mass and
        corresponding magnitudes for each step along the isochrone.
        """
        try:
            columns = self.columns[self.survey.lower()]
        except KeyError as e:
            logger.warning('Unrecognized survey: %s'%(self.survey))
            raise(e)

        kwargs = dict(usecols=list(columns.keys()),dtype=list(columns.values()))
        self.data = np.genfromtxt(filename,**kwargs)
        # cut out anomalous point:
        # https://github.com/DarkEnergySurvey/ugali/issues/29
        self.data = self.data[self.data['stage'] != 9]

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
