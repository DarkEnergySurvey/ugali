#!/usr/bin/env python
"""
Test isochrone functionality. These tests require that ugali has been
installed with the '--isochrones' option.
"""
import os
import glob
import numpy as np

from ugali import isochrone
from ugali.utils.logger import logger
logger.setLevel(logger.WARN)

# Default parameters
default_kwargs = dict(age=12,metallicity=0.0002, distance_modulus=18)
# Alternate parameters
alt_kwargs = dict(age=10, metallicity=0.0001, distance_modulus=16)
# Parameter abbreviations
abbr_kwargs = dict(a=10, z=0.0001, mod=17)

padova = ['Padova','Bressan2012','Marigo2017']
dotter = ['Dotter','Dotter2008','Dotter2016']
isochrones = padova + dotter
survey = ['des','sdss']

# Libraries that are installed by default, and the models that exist for
# each. The LSST, Roman and Euclid libraries are Marigo+ 2017 only.
survey_models = {
    'des'    : ['Bressan2012','Marigo2017','Dotter2008','Dotter2016'],
    'ps1'    : ['Bressan2012','Marigo2017','Dotter2008','Dotter2016'],
    'sdss'   : ['Bressan2012','Marigo2017','Dotter2008','Dotter2016'],
    'lsst'   : ['Bressan2012','Marigo2017','Dotter2016'],
    'roman'  : ['Bressan2012','Marigo2017','Dotter2016'],
    'euclid' : ['Bressan2012','Marigo2017'],
}

# Bands to use for each photometric system
survey_bands = {
    'des'    : ('g','r'),
    'ps1'    : ('g','r'),
    'sdss'   : ('g','r'),
    'lsst'   : ('g','r'),
    'roman'  : ('F106','F158'),
    'euclid' : ('VIS','H'),
}

def has_library(survey, name):
    """ Is the library for this survey and model installed?

    A partial install is normal: only a subset of the libraries is downloaded
    by default, and a library that a release has not published yet will not be
    there at all. Tests that need one skip when it is missing.
    """
    path = os.path.join(isochrone.get_iso_dir(),survey,name.lower())
    if os.path.exists(path): return True
    logger.warning("Library not installed: %s %s"%(survey,name))
    return False

def set_parameters(name):
    iso = isochrone.factory(name,**default_kwargs)

    # Test that parameters are set in construction
    for k,v in default_kwargs.items():
        assert getattr(iso,k) == v

    # Test that parameters are set through setattr
    for k,v in alt_kwargs.items():
        setattr(iso,k,v)
        assert getattr(iso,k) == v

    # Test that parameters are set through setp
    for k,v in default_kwargs.items():
        iso.setp(k,v)
        assert getattr(iso,k) == v

    iso.sample()

def test_exists():
    """ Check that the isochrone directory exists. """
    isodir = isochrone.get_iso_dir()
    assert os.path.exists(isodir)
    assert len(os.listdir(isodir))


def test_abbr(name='Padova'):
    """ Test that parameters can be set by abbreviation. """
    iso = isochrone.factory(name,**abbr_kwargs)

    for k,v in abbr_kwargs.items():
        setattr(iso,k,v)

    for k,v in abbr_kwargs.items():
        iso.setp(k,v)


def test_padova(): 
    for name in padova:
        set_parameters(name)

def test_dotter(): 
    for name in dotter:
        set_parameters(name)

    
def test_composite():
    isochrones = [
        dict(name='Padova',**default_kwargs),
        dict(name='Dotter',**default_kwargs)
    ]
    iso = isochrone.factory("Composite",isochrones=isochrones)

    iso.distance_modulus = alt_kwargs['distance_modulus']
    assert iso.distance_modulus == alt_kwargs['distance_modulus']

    assert np.all(iso.age == np.ones(len(isochrones))*default_kwargs['age'])
    assert np.all(iso.metallicity == np.ones(len(isochrones))*default_kwargs['metallicity'])
    
    iso.sample()

def test_surveys():
    """ Create isochrones with different surveys """
    for s in survey:
        for name in ['Dotter2016']:
            iso = isochrone.factory(name,survey=s)

def test_survey_libraries():
    """ Read every survey/model library of the default install.

    The assertions on the mass range are what catches a column mapping that
    does not match the file format: reading the wrong column silently gives a
    constant 'mass_init' and a mass pdf of zeros.
    """
    for s, names in sorted(survey_models.items()):
        band_1, band_2 = survey_bands[s]
        for name in names:
            if not has_library(s,name): continue
            iso = isochrone.factory(name, survey=s, band_1=band_1,
                                    band_2=band_2, **default_kwargs)
            assert len(iso.mass_init) > 50
            assert iso.mass_init.min() < 0.2 < 0.5 < iso.mass_init.max()
            assert np.all(np.isfinite(iso.mag))
            assert np.sum(iso.sample()[1]) > 0

def test_lsst_bands():
    """ The LSST y band is available as both 'y' and 'Y'. """
    if not has_library('lsst','Marigo2017'): return

    kwargs = dict(default_kwargs, survey='lsst', band_1='g')
    np.testing.assert_array_equal(
        isochrone.factory('Marigo2017', band_2='y', **kwargs).mag_2,
        isochrone.factory('Marigo2017', band_2='Y', **kwargs).mag_2)

def test_vega_to_ab():
    """ Roman isochrones are converted from Vega to AB magnitudes. """
    if not has_library('roman','Marigo2017'): return

    from ugali.isochrone import parsec

    kwargs = dict(default_kwargs, survey='roman', band_1='F106',
                  band_2='F158')
    ab = isochrone.factory('Marigo2017', **kwargs)

    # Read the same file with the conversion disabled
    offsets = parsec.vega_to_ab_dict.pop('roman')
    try:
        vega = isochrone.factory('Marigo2017', **kwargs)
    finally:
        parsec.vega_to_ab_dict['roman'] = offsets

    np.testing.assert_allclose(ab.mag_1 - vega.mag_1, offsets['F106'])
    np.testing.assert_allclose(ab.mag_2 - vega.mag_2, offsets['F158'])

def test_photsys_metadata():
    """ Every photometric system is completely described. """
    from ugali.isochrone.parsec import photsys_dict, photname_dict, bands_dict

    assert set(photsys_dict) == set(photname_dict) == set(bands_dict)

    # 'lsst' is the LSST library that ugali distributes (R1.9 throughputs);
    # 'lsst_r1p9' is retained as an alias for it
    assert photsys_dict['lsst'] == photsys_dict['lsst_r1p9']
    assert photsys_dict['lsst'] != photsys_dict['lsst_dp0']
    assert photsys_dict['lsst'] != photsys_dict['lsst_2012']

def test_column_numbers():
    """ Column numbers are resolved from the file header. """
    if not has_library('lsst','Marigo2017'): return

    from ugali.isochrone.parsec import Marigo2017

    iso = isochrone.factory('Marigo2017', survey='lsst', **default_kwargs)
    columns = Marigo2017._find_column_numbers(iso.filename, 'lsst')
    names = [v[0] for v in columns.values()]
    for name in ['mass_init','mass_act','log_lum','stage','g','r','y']:
        assert name in names

def test_match_band():
    """ Band columns are matched across the CMD naming conventions. """
    from ugali.isochrone.parsec import ParsecIsochrone as P

    # plain ('umag'), suffixed ('gP1mag') and prefixed ('DES-gmag') names
    names = ['Zini','Mini','mbolmag','umag','gmag']
    assert P._match_band('u',names) == 3
    assert P._match_band('Y',names) is None

    names = ['mbolmag','gP1mag','rP1mag','wP1mag']
    assert P._match_band('g',names,'P1') == 1
    assert P._match_band('w',names,'P1') == 3

    names = ['mbolmag','DECam-umag','DES-gmag','DES-Ymag']
    assert P._match_band('u',names) == 1
    assert P._match_band('g',names) == 2
    assert P._match_band('Y',names) == 3

    # an exact match wins over a prefixed one
    names = ['gmag','DES-gmag']
    assert P._match_band('g',names) == 0

def test_mesa_column_numbers():
    """ MIST columns are resolved from the file header.

    The released libraries are MIST v1.0; v1.2 inserted a 'log_R' column that
    shifts the luminosity, the magnitudes and the phase by one. Resolving from
    the header has to reproduce the hard-coded v1.0 numbering exactly on the
    old files, which is what this checks -- if it ever stops doing so, the
    distributed libraries are being read with the wrong columns.
    """
    from ugali.isochrone.mesa import Dotter2016

    for s in ['des','sdss','ps1']:
        if not has_library(s,'Dotter2016'): continue
        path = os.path.join(isochrone.get_iso_dir(),s,'dotter2016')
        filename = sorted(glob.glob(os.path.join(path,'*.dat')))[0]

        columns = Dotter2016._find_column_numbers(filename,s)
        assert columns is not None
        resolved = {(k[0] if isinstance(k,tuple) else k):v[0]
                    for k,v in columns.items()}
        for num,(name,_) in Dotter2016.columns[s].items():
            assert resolved[num].lower() == name.lower()

def test_mesa_band_column():
    """ MIST band columns are matched on the full '<system>_<band>' name.

    'log_g' and 'log_R' come before the magnitudes and would be picked up as
    the LSST 'g' and 'r' bands by a looser match.
    """
    from ugali.isochrone.mesa import Dotter2016

    names = ['EEP','isochrone_age_yr','initial_mass','star_mass','log_Teff',
             'log_R','log_g','log_L','[Fe/H]_init','[Fe/H]',
             'LSST_u','LSST_g','LSST_r','LSST_i','LSST_z','LSST_y','phase']
    assert Dotter2016._band_column('g',names,'lsst') == 11
    assert Dotter2016._band_column('r',names,'lsst') == 12
    assert Dotter2016._band_column('Y',names,'lsst') == 15
    assert Dotter2016._band_column('F062',names,'lsst') is None

    # the prefix is not always the form's 'output' value
    ps1 = ['EEP','isochrone_age_yr','initial_mass','star_mass','log_Teff',
           'log_g','log_L','[Fe/H]_init','[Fe/H]','PS_g','PS_r','PS_w','phase']
    assert Dotter2016._band_column('g',ps1,'ps1') == 9
    assert Dotter2016._band_column('w',ps1,'ps1') == 11

def test_mesa_metadata():
    """ Every MIST photometric system is completely described. """
    from ugali.isochrone.mesa import dict_output, band_prefix_dict, bands_dict

    assert set(dict_output) == set(band_prefix_dict) == set(bands_dict)

def test_ab_magnitudes():
    """ Only PARSEC's Roman tables need a Vega->AB conversion.

    MIST serves every system in AB, so applying the PARSEC offsets to a MIST
    Roman isochrone would double-count them.
    """
    from ugali.isochrone.parsec import Marigo2017
    from ugali.isochrone.mesa import Dotter2016

    assert 'roman' in Marigo2017.vega_to_ab
    assert Dotter2016.vega_to_ab == {}

def test_import():
    """ Test various import strategies """
    import ugali.analysis.isochrone
    from ugali.analysis.isochrone import Bressan2012, CompositeIsochrone

    import ugali.isochrone
    from ugali.isochrone import Bressan2012, CompositeIsochrone

def test_pdf():
    """ 
    Test the isochrone.pdf function.  

    This test should use ~300 MiB of memory...
    """
    iso = isochrone.Bressan2012(**default_kwargs)
    mag_1,mag_2 = np.meshgrid(np.linspace(18,22,100),np.linspace(18,22,100))
    mag_1 = mag_1.flatten()
    mag_2 = mag_2.flatten()
    mag_err_1 = 0.1 * np.ones_like(mag_1)
    mag_err_2 = 0.1 * np.ones_like(mag_2)
    u_color = iso.pdf(mag_1, mag_2, mag_err_1, mag_err_2)
    test_results = np.array([0.00103531, 0.00210507, 0.00393214, 0.00675272, 
                             0.01066913, 0.01552025, 0.0208020, 0.02570625, 
                             0.02930542, 0.03083482], dtype=np.float32)
    np.testing.assert_array_almost_equal(u_color[9490:9500],test_results)

def test_simulate():
    """Test isochrone simulation."""    
    
    iso = isochrone.Bressan2012(**default_kwargs)
    stellar_mass = 5.0e3

    np.random.seed(0)
    mag_1, mag_2 = iso.simulate(stellar_mass)
    np.testing.assert_equal(len(mag_1), 21106)
    np.testing.assert_allclose(mag_1[:3], [28.606918, 27.670816, 28.302291])
    np.testing.assert_allclose(mag_2[:3], [27.539174, 26.717612, 27.271779])

def test_download():
    """Test isochrone download."""
    try:
        from urllib.error import URLError
    except ImportError:
        from urllib2 import URLError

    for name in dotter[1:]: #padova[1:]+dotter[1:]:
        iso = isochrone.factory(name,**default_kwargs)
        try:
            iso.download(outdir='./tmp/'+name.lower(),force=True)
        except URLError as e:
            logger.error("%s: Server is down.\n%s"%(name,str(e)))
        except RuntimeError as e:
            logger.error("%s: %s"%(name,str(e)))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    
