#!/usr/bin/env python
"""
MESA/MIST Isochrones from:
https://mist.science/interp_isos.html
"""
import os
import sys
import glob
import copy
import re
import zipfile
import tempfile
import shutil
import contextlib
from collections import OrderedDict as odict

try:
    from urllib.parse import urlencode
    from urllib.request import urlopen, Request
except ImportError:
    from urllib import urlencode
    from urllib2 import urlopen, Request

import numpy as np

from ugali.utils.logger import logger

from ugali.isochrone.parsec import Isochrone
from ugali.isochrone.model import get_iso_dir

###########################################################
# MESA Isochrones
# https://mist.science/interp_isos.html
#
# NOTE: the MIST web interface used to live at
# http://waps.cfa.harvard.edu/MIST, which now redirects to https://mist.science
# and no longer answers the old form. The move also renamed some of the form
# fields and the output file, see `Dotter2016.query_server`.

# Photometric system, as the 'output' value of the MIST form
dict_output = odict([
        ('des','DECam'),
        ('sdss','SDSSugriz'),
        ('ps1','PanSTARRS'),
        ('lsst','LSST'),
        ('roman','Roman'),
])

# Prefix that MIST puts on the magnitude columns of each photometric system.
# This is *not* always the 'output' value of the form ('SDSSugriz' produces
# 'SDSS_u', 'PanSTARRS' produces 'PS_g'), so it is tabulated separately.
band_prefix_dict = odict([
        ('des','DECam'),
        ('sdss','SDSS'),
        ('ps1','PS'),
        ('lsst','LSST'),
        ('roman','Roman'),
])

# Bands of each photometric system. Used to resolve the magnitude columns
# from the file header; see Isochrone._find_column_numbers.
bands_dict = odict([
        ('des' ,['u','g','r','i','z','Y']),
        ('sdss',['u','g','r','i','z']),
        ('ps1' ,['g','r','i','z','y','w']),
        ('lsst',['u','g','r','i','z','y']),
        ('roman',['F062','F087','F106','F129','F146','F158','F184','F213']),
])

# NOTE: MIST serves every photometric system in AB magnitudes -- the header
# of each file says so explicitly ('LSST (AB)', 'Roman (AB)') -- so unlike
# the CMD/PARSEC Roman tables no Vega->AB conversion is needed here.

mesa_defaults = {
        'version':'MIST1',   # 'MIST1' = v1.2, 'MIST2' = v2.5
        'v_div_vcrit':'vvcrit0.4',
        'age_scale':'linear',
        'age_type':'single',
        'age_value':10e9, # yr if scale='linear'; log10(yr) if scale='log10'
        'age_range_low':'',
        'age_range_high':'',
        'age_range_delta':'',
        'age_list':'',
        'FeH_value':-3.0,
        'alpha_value':'p0', # [a/Fe]; 'p0' is scaled-solar
        'output_option':'photometry',
        'output':'DECam',
        'Av_value':0,
}

mesa_defaults_10 = dict(mesa_defaults,version='MIST1')

class Dotter2016(Isochrone):
    """ MESA isochrones from Dotter 2016:
    https://mist.science/interp_isos.html
    """
    _dirname =  os.path.join(get_iso_dir(),'{survey}','dotter2016')

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage',3,'Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    download_url = 'https://mist.science'
    download_defaults = copy.deepcopy(mesa_defaults_10)

    abins = np.arange(1., 13.5+0.1, 0.1)
    zbins = np.arange(1e-5, 1e-3+1e-5, 1e-5)

    # Map from the ugali column names to the MIST header names.
    header_names = odict([
            ('mass_init', ['initial_mass']),
            ('mass_act' , ['star_mass']),
            ('log_lum'  , ['log_L']),
            ('stage'    , ['phase']),
            ])

    band_names = bands_dict

    # Legacy column numbers, correct for the MIST v1.0 files that the
    # released DES/PS1/SDSS libraries contain. They are *not* correct for
    # anything the current form returns: MIST v1.2 inserted a 'log_R' column
    # at index 5, shifting the luminosity, the magnitudes and the phase by
    # one. Files with a column header are resolved from that header instead
    # (see Isochrone._find_column_numbers); these are only the fallback for
    # a file that has none. There is deliberately no 'lsst' or 'roman' entry:
    # no v1.0 file exists for either, so a headerless file of those surveys
    # should fail loudly rather than be read with the wrong columns.
    columns = dict(
            des = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (6, ('log_lum',float)),
                (9, ('u',float)),
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

    @classmethod
    def _header_columns(cls, filename):
        """ Column names from the header of a MIST photometry file.

        MIST writes the column names on the last comment line of the file,
        after a line numbering the columns.
        """
        names = None
        with open(filename,'r') as f:
            for line in f:
                if not line.startswith('#'): break
                tokens = line.lstrip('#').split()
                if 'EEP' in tokens and 'initial_mass' in tokens:
                    names = tokens
        return names

    @classmethod
    def _band_column(cls, band, names, survey):
        """ Column of a band among the MIST header column names.

        MIST names its magnitude columns '<system>_<band>' ('LSST_g',
        'DECam_Y', 'PS_g'). The match is deliberately exact rather than a
        suffix match: several of the theory columns end in something that
        looks like a band name ('log_g' and 'log_R' would otherwise be
        picked up as the LSST 'g' and 'r' bands, and they come first).
        """
        prefix = band_prefix_dict.get(survey.lower())
        if prefix is None: return None
        target = ('%s_%s'%(prefix,band)).lower()
        lower = [n.lower() for n in names]
        return lower.index(target) if target in lower else None

    def _parse(self,filename):
        """
        Reads an isochrone in the Dotter 2016 format and determines
        the age (Gyr), metallicity (Z), and creates arrays with the
        initial stellar mass and corresponding magnitudes for each
        step along the isochrone.
        """
        columns = self._find_column_numbers(filename,self.survey)
        if columns is not None:
            kwargs = self._genfromtxt_kwargs(columns)
            kwargs['comments'] = '#'
        else:
            # A file with no column header; fall back to the v1.0 numbering
            try:
                columns = self.columns[self.survey.lower()]
            except KeyError as e:
                logger.warning('Unrecognized survey: %s'%(self.survey))
                raise(e)
            kwargs = dict(comments='#',usecols=list(columns.keys()),
                          dtype=list(columns.values()))

        data = self._read_data(filename,**kwargs)

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
        """ Download one isochrone from the MIST web interface.

        NOTE: the interface moved from http://waps.cfa.harvard.edu/MIST to
        https://mist.science, which changed three things: the zip is served
        out of 'output/' rather than 'tmp/', the photometry file inside it is
        named '<basename>.iso.<output>' rather than '<basename>.cmd', and the
        form fields changed ('version' now takes 'MIST1'/'MIST2' rather than
        a bare version number, 'theory_output' is gone and 'alpha_value' is
        new).
        """
        z = metallicity
        feh = self.z2feh(z)

        params = dict(self.download_defaults)
        params['output'] = dict_output[self.survey]
        params['FeH_value'] = feh
        params['age_value'] = age * 1e9
        if params['age_scale'] == 'log10':
            params['age_value'] = np.log10(params['age_value'])

        server = self.download_url
        url = server + '/iso_form.php'
        logger.debug("Accessing %s..."%url)

        q = urlencode(params).encode('utf-8')
        request = Request(url,data=q)
        response = urlopen(request).read().decode('utf-8',errors='replace')

        # The response is a single link to the zipped output
        match = re.search(r'href="([^"]+\.zip)"',response)
        if match is None:
            logger.debug(response)
            msg = 'Output filename not found'
            raise RuntimeError(msg)
        href = match.group(1)

        tmpdir = tempfile.mkdtemp()
        try:
            zipname = os.path.join(tmpdir,os.path.basename(href))
            zipurl = '{0}/{1}'.format(server,href.lstrip('/'))
            logger.debug("Downloading %s..."%zipurl)
            with contextlib.closing(urlopen(zipurl)) as response:
                with open(zipname,'wb') as tmp:
                    shutil.copyfileobj(response,tmp)

            with zipfile.ZipFile(zipname) as zf:
                # The photometry file is '<basename>.iso.<output>'; the plain
                # '.iso' alongside it holds the theory quantities only.
                suffix = '.iso.%s'%params['output']
                members = [n for n in zf.namelist() if n.endswith(suffix)]
                if not members:
                    msg = "No '%s' file in %s"%(suffix,os.path.basename(href))
                    raise RuntimeError(msg)
                zf.extract(members[0],tmpdir)
                logger.debug("Creating %s..."%outfile)
                shutil.move(os.path.join(tmpdir,members[0]),outfile)
        finally:
            shutil.rmtree(tmpdir,ignore_errors=True)

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
                a = float(lines[13].split()[1])
                # The age column is named 'isochrone_age_yr' in every MIST
                # version, but v1.0 writes the age in years while v1.2 writes
                # log10(age/yr). Accept either rather than trusting the name.
                assert (np.allclose(age,a,atol=1e-5) or
                        np.allclose(np.log10(age),a,atol=1e-5))
            except:
                msg = "Age does not match:\n"+lines[13]
                raise Exception(msg)
