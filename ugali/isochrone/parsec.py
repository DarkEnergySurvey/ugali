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

import shutil
import contextlib
import re

import numpy as np
import scipy.spatial

from ugali.utils.logger import logger
from ugali.isochrone.model import Isochrone
from ugali.isochrone.model import get_iso_dir

# survey system
#
# Keys are the ugali survey names, values are the photometric system files
# served by the CMD web interface. Note that 'lsst' is the current LSST total
# throughput set (R1.9, Sept 2023); these are the LSST isochrones that ugali
# distributes. The older throughput sets remain available under explicit
# names: 'lsst_dp0' (Oct 2017, used for the DP0/DC2 simulations) and
# 'lsst_2012' (March 2012, the original tab_mag_lsst.dat). 'lsst_r1p9' is
# retained as a deprecated alias for 'lsst'.
photsys_dict = odict([
        ('des' ,'tab_mag_odfnew/tab_mag_decam.dat'),
        ('sdss','tab_mag_odfnew/tab_mag_sloan.dat'),
        ('ps1' ,'tab_mag_odfnew/tab_mag_panstarrs1.dat'),
        ('acs_wfc' ,'tab_mag_odfnew/tab_mag_acs_wfc.dat'),
        ('lsst', 'tab_mag_odfnew/tab_mag_lsstR1.9.dat'),
        ('lsst_r1p9', 'tab_mag_odfnew/tab_mag_lsstR1.9.dat'),
        ('lsst_dp0', 'tab_mag_odfnew/tab_mag_lsstDP0.dat'),
        ('lsst_2012', 'tab_mag_odfnew/tab_mag_lsst.dat'),
        ('roman', 'tab_mag_odfnew/tab_mag_Roman2021.dat'),
        ('euclid', 'tab_mag_odfnew/tab_mag_euclid_nisp.dat'),
])

photname_dict = odict([
        ('des' ,'DECAM'),
        ('sdss','SDSS'),
        ('ps1' ,'Pan-STARRS1'),
        ('acs_wfc','HST/ACS'),
        ('lsst', 'LSST'),
        ('lsst_r1p9', 'LSST'),
        ('lsst_dp0', 'LSST'),
        ('lsst_2012', 'LSST'),
        ('roman', 'Roman'),
        ('euclid', 'Euclid'),
])

# Bands of each photometric system. These are used to resolve the magnitude
# columns from the names in the file header (see
# ParsecIsochrone._find_column_numbers), which is more robust than the
# hard-coded column numbers in `Isochrone.columns` because the CMD column
# layout has changed between versions. Single-character band names are also
# made available in the opposite case (i.e. both 'y' and 'Y').
bands_dict = odict([
        ('des' ,['u','g','r','i','z','Y']),
        ('sdss',['u','g','r','i','z']),
        ('ps1' ,['g','r','i','z','y','w']),
        ('acs_wfc',['F435W','F475W','F555W','F606W','F625W','F775W','F814W']),
        ('lsst', ['u','g','r','i','z','y']),
        ('lsst_r1p9', ['u','g','r','i','z','y']),
        ('lsst_dp0', ['u','g','r','i','z','y']),
        ('lsst_2012', ['u','g','r','i','z','y']),
        ('roman', ['F062','F087','F106','F129','F158','F184','F146','F213']),
        ('euclid', ['VIS','Y','Blue','J','Red','H']),
])

# CMD names its magnitude columns '<band>mag', but not uniformly: the
# Pan-STARRS1 columns carry a filter-set suffix ('gP1mag'). Prefixed names
# (the current DECam table uses 'DES-gmag' and 'DECam-umag') are handled
# without any per-survey configuration, see ParsecIsochrone._match_band.
band_suffix_dict = odict([
        ('ps1','P1'),
])

# Photometric systems that the CMD server returns in Vega magnitudes. ugali
# works in AB magnitudes, so these offsets (m_AB - m_Vega) are applied when
# the isochrone is parsed. The Roman values come from Roman-STScI-000825 and
# match the conversion that was previously applied downstream in
# LSSTDESC/streamobs.
vega_to_ab_dict = odict([
        ('roman', odict([
                ('F062',0.153), ('F087',0.481), ('F106',0.660),
                ('F129',1.051), ('F146',1.164), ('F158',1.315),
                ('F184',1.556), ('F213',1.837),
                ])),
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

defaults_36 = dict(defaults_33,cmd_version=3.6)
defaults_39 = dict(defaults_33,cmd_version=3.9)


class ParsecIsochrone(Isochrone):
    """ Base class for PARSEC-style isochrones. """

    # NOTE: must be https. The server 301-redirects http -> https, and
    # urllib downgrades a redirected POST to a GET, so the query is dropped
    # and the response is the blank form rather than an isochrone -- which
    # surfaces as the unhelpful 'Output filename not found'.
    download_url = "https://stev.oapd.inaf.it"
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
        logger.debug("%s?%s"%(url,q))
        c = str(urlopen(url, q).read())
        aa = re.compile(r'output\d+')
        fname = aa.findall(c)
        
        if len(fname) == 0:
            msg = "Output filename not found"
            raise RuntimeError(msg)

        out = '{0}/tmp/{1}.dat'.format(server, fname[0])

        # NOTE: fetched with urlopen rather than by shelling out to wget, so
        # that the download uses the same TLS configuration as the query
        # above. The CMD server has been seen to serve an incomplete
        # certificate chain, and the usual fix (pointing SSL_CERT_FILE at a
        # bundle carrying the missing intermediate) reaches Python but not a
        # wget subprocess, which would fail the download after a successful
        # query. It also drops a shell dependency.
        logger.debug("Downloading %s..."%out)
        with contextlib.closing(urlopen(out)) as response:
            with open(outfile,'wb') as tmp:
                shutil.copyfileobj(response,tmp)

        return outfile

    # Map from the ugali column names to the names used in the header of a
    # modern (cmd_3.3 and later) CMD file.
    header_names = odict([
            ('mass_init', ['Mini']),
            ('mass_act' , ['Mass']),
            ('log_lum'  , ['logL']),
            ('stage'    , ['label']),
            ])

    # Bands and Vega->AB offsets for the CMD photometric systems. These are
    # module-level so that they can be extended without subclassing; the
    # class attributes are what the shared read path in Isochrone uses.
    band_names = bands_dict
    vega_to_ab = vega_to_ab_dict
    header_dtypes = odict([('stage',int)])

    @classmethod
    def _header_columns(cls, filename):
        """ Column names from the header of a CMD file.

        Returns None for the legacy cmd_2.7 format, which does not name its
        columns and has to fall back to the hard-coded `columns`.
        """
        names = None
        with open(filename,'r') as f:
            for line in f:
                if not line.startswith('#'): break
                tokens = line.lstrip('#').split()
                # The column line of a modern file names the magnitudes with
                # a 'mag' suffix ('mbolmag', 'gmag', ...); the legacy format
                # names them 'mbol', 'g', ... and is not self-describing.
                if any(t.endswith('mag') for t in tokens) and 'Mini' in tokens:
                    names = tokens
        return names

    @classmethod
    def _band_column(cls, band, names, survey):
        """ Column of a band, using this survey's filter-set suffix. """
        return cls._match_band(band,names,
                               band_suffix_dict.get(survey.lower(),''))

    @staticmethod
    def _match_band(band, names, suffix=''):
        """ Find the column of a band among the CMD header column names.

        The magnitude columns are named '<band>mag', but the band can carry a
        filter-set suffix ('gP1mag' for Pan-STARRS1) or a prefix ('DES-gmag'
        and 'DECam-umag' in the current DECam table). Matches are tried from
        most to least specific.

        Parameters
        ----------
        band   : the ugali band name
        names  : the column names from the file header
        suffix : filter-set suffix for this photometric system

        Returns
        -------
        index : column number of the band, or None if it is not present
        """
        cores = [n[:-3].lower() if n.lower().endswith('mag') else None
                 for n in names]

        for candidate in [band.lower(), (band+suffix).lower()]:
            if candidate in cores: return cores.index(candidate)

        for i,core in enumerate(cores):
            if core and core.split('-')[-1] == band.lower(): return i

        return None

    @classmethod
    def parse_header(cls, filename, nlines=15):
        header = dict(
            photname = None,
            columns = None
        )

        with open(filename,'r') as f:
            lines = [f.readline() for i in range(nlines)]

        if len(lines) < nlines:
            msg = "Incorrect file size"
            raise Exception(msg)

        for i,l in enumerate(lines):
            if l.startswith('# Photometric system:'): 
                try:    header['photname'] = lines[i].split()[3]
                except: header['photname'] = None
            if not l.startswith('# '): break

        header['columns'] = lines[i-1].split()[1:]

        for k,v in header.items():
            if v is None: 
                msg = "File header missing: '%s'"%k
                raise Exception(msg)

        return header, lines
    
    @classmethod
    def verify(cls, filename, survey, age, metallicity):
        """Verify that the isochrone file matches the isochrone
        parameters. Used mostly for verifying the integrity of 
        a download.

        Parameters
        ----------
        filename    : the downloaded filename 
        survey      : the survey (photometric system) of the download
        age         : the requested age of the system
        metallicity : the requested metallicity

        Returns
        -------
        None
        """
        age = age*1e9
        nlines=15
        header, lines = cls.parse_header(filename,nlines=nlines)

        try:
            assert photname_dict[survey] == header['photname']
        except:
            msg = "Incorrect survey:\n"+header['photname']
            raise Exception(msg)

        # A fresh download always comes from a modern CMD version, so the
        # columns must be resolvable from the header (see #104).
        if cls._find_column_numbers(filename,survey) is None:
            msg = "Unrecognized column format:\n"+header['columns'][0]
            raise Exception(msg)

        try:
            try: zidx = header['columns'].index('Zini')
            except ValueError: zidx = 0
            z = float(lines[-1].split()[zidx])
            assert np.allclose(metallicity,z,atol=1e-5)
        except:
            msg = "Metallicity does not match:\n"+lines[-1]
            raise Exception(msg)

        try:
            # Need to deal with age or log-age
            names = ['log(age/yr)', 'logAge', 'Age']
            for name in names:
                try: 
                    aidx = header['columns'].index(name)
                    break
                except ValueError: 
                    aidx = 1
            if aidx < 0: aidx = 1

            a = lines[-1].split()[aidx]
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

    # The cmd_3.1 form no longer returns data for the cmd_2.7-era parameter
    # set, so query the current interface instead. PARSEC v1.2S without
    # COLIBRI (i.e. Bressan+ 2012) is selected with track_colibri='no';
    # 'isoc_kind' is only read by cmd_2.7/3.1 and is kept for reference.
    download_defaults = copy.deepcopy(defaults_39)
    download_defaults['track_colibri'] = 'no'
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
        columns = self._find_column_numbers(filename,self.survey)
        if columns is not None:
            # Modern (cmd_3.3+) whitespace-delimited file with named columns
            kwargs = self._genfromtxt_kwargs(columns)
        else:
            # Legacy (cmd_2.7/3.1) file; fall back to hard-coded columns.
            # delimiter='\t' is used to be compatible with OldPadova...
            # ADW: This should be updated, but be careful of column numbering
            try:
                columns = self.columns[self.survey.lower()]
            except KeyError as e:
                logger.warning('Unrecognized survey: %s'%(self.survey))
                raise(e)
            kwargs = dict(delimiter='\t',usecols=list(columns.keys()),
                          dtype=list(columns.values()))

        self._read_data(filename,**kwargs)

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

    # PARSEC v1.2S + COLIBRI (i.e. Marigo+ 2017) is the default track_colibri
    # of defaults_39; 'isoc_kind' is only read by cmd_2.7/3.1.
    download_defaults = copy.deepcopy(defaults_39)
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
        lsst_dp0 = odict([
                (3, ('mass_init',float)),
                (5, ('mass_act',float)),
                (6, ('log_lum',float)),
                (9, ('stage',int)),
                (25,('u',float)),
                (26,('g',float)),
                (27,('r',float)),
                (28,('i',float)),
                (29,('z',float)),
                (30,('Y',float)),
                ]),
        roman = odict([
                (3, ('mass_init',float)),
                (5, ('mass_act',float)),
                (6, ('log_lum',float)),
                (9,('stage',float)),
                (25, ('F062',float)),
                (26,('F087',float)),
                (27,('F106',float)),
                (28,('F129',float)),
                (29,('F158',float)),
                (30,('F184',float)),
                (31,('F146',float)),
                (32,('F213',float)),
                ]),
        euclid = odict([
                (3, ('mass_init',float)),
                (5, ('mass_act',float)),
                (6, ('log_lum',float)),
                (9,('stage',float)),
                (25, ('VIS',float)),
                (26,('Y',float)),
                (28,('J',float)),
                (30,('H',float)),
                ]),
        )
    columns['lsst'] = copy.deepcopy(columns['lsst_dp0'])
    columns['lsst_r1p9'] = copy.deepcopy(columns['lsst_dp0'])

    def _parse(self,filename):
        """Reads an isochrone file in the Marigo et al. 2017
        format. Creates arrays with the initial stellar mass and
        corresponding magnitudes for each step along the isochrone.

        Parameters:
        -----------
        filename : name of isochrone file to parse

        Returns:
        --------
        None
        """
        columns = self._find_column_numbers(filename,self.survey)
        if columns is not None:
            kwargs = self._genfromtxt_kwargs(columns)
        else:
            try:
                columns = self.columns[self.survey.lower()]
            except KeyError as e:
                logger.warning('Unrecognized survey: %s'%(self.survey))
                raise(e)
            kwargs = dict(usecols=list(columns.keys()),
                          dtype=list(columns.values()))

        self._read_data(filename,**kwargs)
        # cut out anomalous point:
        # https://github.com/DarkEnergySurvey/ugali/issues/29
        self.data = self.data[~np.isin(self.data['stage'], [9])]

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
