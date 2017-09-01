#!/usr/bin/env python
"""
Module for dealing with Isochrones.
"""
from ugali.isochrone.model import get_iso_dir
from ugali.isochrone.parsec import Bressan2012, Marigo2017
from ugali.isochrone.dartmouth import Dotter2008
from ugali.isochrone.mesa import Dotter2016

# ADW: It'd be better for us to move away from generic aliases...
Dotter = Dotter2016
Padova = Marigo2017

def factory(name, **kwargs):
    from ugali.utils.factory import factory

    # First try here
    try:    return factory(name, module=__name__, **kwargs)
    except: pass
    # Then try parsec
    try:    return factory(name, module=__name__+'.parsec', **kwargs)
    except: pass
    # Then try mesa
    try:    return factory(name, module=__name__+'.mesa', **kwargs)
    except: pass
    # Then try desd
    try:    return factory(name, module=__name__+'.dartmouth', **kwargs)
    except: pass
        
    raise KeyError('Unrecognized class: %s'%name)
