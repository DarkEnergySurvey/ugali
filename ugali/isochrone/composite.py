#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

import sys
import os
from collections import OrderedDict as odict
from functools import wraps

import numpy as np

from ugali.analysis.model import Model, Parameter
from ugali.utils.logger import logger
from ugali.isochrone.model import IsochroneModel, Isochrone
from ugali.isochrone.mesa import Dotter2016
from ugali.isochrone.parsec import Marigo2017

class CompositeIsochrone(IsochroneModel):
    _params = odict([
        ('distance_modulus', Parameter(15.0, [10.0, 30.0]) ),
    ])
    _mapping = odict([
            ('mod','distance_modulus'),
            ('a','age'),                 
            ('z','metallicity'),
            ])

    defaults = (IsochroneModel.defaults) + (
        ('type','PadovaIsochrone','Default type of isochrone to create'),
        ('weights',None,'Relative weights for each isochrone'),
        )
    
    def __init__(self, isochrones, **kwargs):
        super(CompositeIsochrone,self).__init__(**kwargs)

        self.isochrones = []
        for i in isochrones:
            if isinstance(i,Isochrone):
                iso = i
            else:
                name = i.pop('name',self.type)
                #iso = isochroneFactory(name=name,**i)
                iso = factory(name=name,**i)
            # Tie the distance modulus
            iso.params['distance_modulus'] = self.params['distance_modulus']
            self.isochrones.append(iso)
        
        if self.weights is None: self.weights = np.ones(len(self.isochrones))
        self.weights /= np.sum(np.asarray(self.weights))
        self.weights = self.weights.tolist()

        if len(self.isochrones) != len(self.weights):
            msg = 'Length of isochrone and weight arrays must be equal'
            raise ValueError(msg)

    def __getitem__(self, key):
        return self.isochrones[key]
 
    def __str__(self,indent=0):
        ret = super(CompositeIsochrone,self).__str__(indent)
        ret += '\n{0:>{2}}{1}'.format('','Isochrones:',indent+2)
        for i in self:
            ret += '\n{0}'.format(i.__str__(indent+4))
        return ret

    @property
    def age(self):
        return np.array([i.age for i in self])

    @property
    def metallicity(self):
        return np.array([i.metallicity for i in self])

    def composite_decorator(func):
        """
        Decorator for wrapping functions that calculate a weighted sum
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            total = []
            for weight,iso in zip(self.weights,self.isochrones):
                subfunc = getattr(iso,func.__name__)
                total.append(weight*subfunc(*args,**kwargs))
            return np.sum(total,axis=0)
        return wrapper

    def sample(self, **kwargs):
        samples = [iso.sample(**kwargs) for iso in self.isochrones]
        for weight,sample in zip(self.weights,samples):
            sample[1] *= weight

        return np.hstack(samples)

    def separation(self, *args, **kwargs):
        separations = [iso.separation(*args,**kwargs) for iso in self.isochrones]
        return np.nanmin(separations,axis=0)
    
    def todict(self):
        ret = super(CompositeIsochrone,self).todict()
        ret['isochrones'] = [iso.todict() for iso in self.isochrones]
        return ret

    @composite_decorator
    def stellar_mass(self, *args, **kwargs): pass

    @composite_decorator
    def stellar_luminosity(self, *args, **kwargs): pass

    @composite_decorator
    def observable_fraction(self, *args, **kwargs): pass

    @composite_decorator
    def observableFractionX(self, *args, **kwargs): pass

    @composite_decorator
    def signalMMD(self, *args, **kwargs): pass

    # ADW: For temporary backwards compatibility
    stellarMass = stellar_mass
    stellarLuminosity = stellar_luminosity
    observableFraction = observable_fraction

# ADW: It would be better if the factory were in isochrone.__init__
# but then you get into a circular import situation with the
# CompositeIsochrone. This is an unfortunate design decision...

# ADW: It'd be better for us to move away from generic aliases...
class Dotter(Dotter2016): pass
class Padova(Marigo2017): pass
class Composite(CompositeIsochrone): pass

def factory(name, **kwargs):
    from ugali.utils.factory import factory

    module = 'ugali.isochrone'
    # First try this module
    #try:    return factory(name, module=__name__, **kwargs)
    #except KeyError: pass
    try:    return factory(name, module=module+'.composite', **kwargs)
    except KeyError: pass
    # Then try parsec
    try:    return factory(name, module=module+'.parsec', **kwargs)
    except KeyError: pass
    # Then try mesa
    try:    return factory(name, module=module+'.mesa', **kwargs)
    except KeyError: pass
    # Then try desd
    try:    return factory(name, module=module+'.dartmouth', **kwargs)
    except KeyError: pass

    raise KeyError('Unrecognized class: %s'%name)
