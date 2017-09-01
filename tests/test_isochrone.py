#!/usr/bin/env python
"""
Test isochrone functionality. These tests require that ugali has been
installed with the '--isochrones' option.
"""
import os
import numpy as np

import ugali.analysis.isochrone as isochrone


# Default parameters
default_kwargs = dict(age=12,metallicity=0.0002, distance_modulus=18)
# Alternate parameters
alt_kwargs = dict(age=13, metallicity=0.0008, distance_modulus=16)
# Parameter abbreviations
abbr_kwargs = dict(a=12.5, z=0.0004, mod=17)

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
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    
