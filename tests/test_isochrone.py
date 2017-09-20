#!/usr/bin/env python
"""
Test isochrone functionality. These tests require that ugali has been
installed with the '--isochrones' option.
"""
import os
import numpy as np

from ugali import isochrone

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    
