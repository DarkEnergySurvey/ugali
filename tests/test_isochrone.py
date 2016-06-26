#!/usr/bin/env python
"""
Test isochrone functionality
"""
import numpy as np

import ugali.analysis.isochrone as isochrone

default_kwargs = dict(age = 12,metallicity = 0.0002, distance_modulus = 18)
test_kwargs = dict(age = 13, metallicity = 0.0008, distance_modulus = 16)

def set_parameters(name):
    iso = isochrone.factory(name,**default_kwargs)
    for k,v in default_kwargs.items():
        assert getattr(iso,k) == v

    for k,v in test_kwargs.items():
        setattr(iso,k,v)
        assert getattr(iso,k) == v

    for k,v in default_kwargs.items():
        iso.setp(k,v)
        assert getattr(iso,k) == v

    iso.sample()

def test_padova(): set_parameters('Padova')
def test_dotter(): set_parameters('Dotter')

def test_composite():
    isochrones = [
        dict(name='Padova',**default_kwargs),
        dict(name='Dotter',**default_kwargs)
    ]
    iso = isochrone.factory("Composite",isochrones=isochrones)

    iso.distance_modulus = test_kwargs['distance_modulus']
    assert iso.distance_modulus == test_kwargs['distance_modulus']

    assert np.all(iso.age == np.ones(len(isochrones))*default_kwargs['age'])
    assert np.all(iso.metallicity == np.ones(len(isochrones))*default_kwargs['metallicity'])
    
    iso.sample()
