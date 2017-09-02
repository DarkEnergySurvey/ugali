#!/usr/bin/env python
import os
import yaml
from collections import OrderedDict as odict
import numpy as np
import copy

from ugali.analysis.model import Model, Parameter
from ugali.analysis.kernel import factory as kernelFactory
from ugali.isochrone import factory as isochroneFactory

class Richness(Model):
    """Dummy model to hold the richness, which is not directly connected
    to either the spatial or color information and doesn't require a
    sync when updated.
    """
    _params = odict([
        ('richness', Parameter(1000.0, [0.0,  np.inf])),
    ])

# Just to be consistent with other factories
def richnessFactory(type='Richness',**kwargs):
    return Richness(**kwargs)

class Source(object):
    """ 
    A source model builds the components on the spatial and spectral model.

    ADW: We could probably be smarter and get this to inherit from Model...
    """
    _defaults = odict([
            ('richness' ,dict(name='Richness')),
            ('kernel'   ,dict(name='Plummer')),
            ('isochrone',dict(name='Padova')),
            ])

    def __init__(self,name=None, **kwargs):
        self.set_model('richness',self.createRichness())
        self.set_model('kernel',self.createKernel())
        self.set_model('isochrone',self.createIsochrone())

        # Toggle for tracking which models need to be synched
        self._sync = odict([(k,True) for k in self.models.keys()])

        self.name = name
        self.set_params(**kwargs)

    def __str__(self):
        ret = "%s : Name=%s"%(self.__class__.__name__,self.name)
        for key,model in self.models.items():
            ret += "\n  %s Model (sync=%s):\n"%(key.capitalize(),self._sync[key])
            ret += model.__str__(indent=4)
        return ret

    def __getattr__(self, name):
        #for key,model in self.models.items():
        #    if name in model.params:
        #        return getattr(model, name)
        #for key,model in self.models.items():
        #    try:
        #        return model.getp(name).value
        #    except KeyError:
        #        continue
        ## Raises AttributeError
        #return object.__getattribute__(self,name)
        try:
            return self.getp(name)
        except AttributeError:
            return object.__getattribute__(self,name)

    def __setattr__(self, name, value):
        #for key,model in self.models.items():
        #    if name in model.params:
        #        self._sync[key] = True
        #        return setattr(model, name, value)
        #for key,model in self.models.items():
        #    try:
        #        ret = model.setp(name, value)
        #        self._sync[key] = True
        #        return ret
        #    except KeyError:
        #        continue
        ## Raises AttributeError?
        #return object.__setattr__(self, name, value)
        try:
            return self.setp(name,value)
        except AttributeError:
            return object.__setattr__(self, name, value)

    def setp(self, name, *args, **kwargs):
        for key,model in self.models.items():
            try:
                ret = model.setp(name, *args, **kwargs)
                self._sync[key] = True
                return ret
            except KeyError:
                continue
        raise AttributeError

    def getp(self, name):
        for key,model in self.models.items():
            try:
                return model.getp(name).value
            except KeyError:
                continue
        raise AttributeError

    @property
    def params(self):
        # DANGEROUS: Altering properties directly doesn't call model._cache
        params = odict([])
        for key,model in self.models.items():
            params.update(model.params)
        return params


    def load(self,srcmdl,section=None):
        if isinstance(srcmdl,basestring): 
            params = yaml.load(open(srcmdl))
        else:
            params = copy.deepcopy(srcmdl)

        if section is not None: 
            params = params[section]
        elif len(params) == 1:
            section = params.keys()[0]
            params = params[section]

        fill = False
        if params.get('name'):
            self.name = params.get('name')
        if params.get('richness'):
            richness  = self.createRichness(**params['richness'])
            self.set_model('richness',richness)
            fill = True
        if params.get('kernel'):
            kernel    = self.createKernel(**params['kernel'])
            self.set_model('kernel',kernel)
            fill = True
        if params.get('isochrone'):
            isochrone = self.createIsochrone(**params['isochrone'])
            self.set_model('isochrone',isochrone)
            fill = True

        if not fill:
            msg = "Didn't load any source parameters."
            raise Exception(msg)

    def dump(self):
        return yaml.dump(self.todict())

    def todict(self):
        ret = odict()
        if self.name is not None: ret['name'] = self.name
        for name,model in self.models.items():
            ret[name] = model.todict()
        return ret
        
    # ADW: Should be class methods
    @staticmethod
    def createRichness(**kwargs):
        for k,v in copy.deepcopy(Source._defaults['richness']).items():
            kwargs.setdefault(k,v)
        return richnessFactory(**kwargs)

    @staticmethod
    def createKernel(**kwargs):
        for k,v in copy.deepcopy(Source._defaults['kernel']).items():
            kwargs.setdefault(k,v)
        return kernelFactory(**kwargs)

    @staticmethod
    def createIsochrone(**kwargs):
        for k,v in copy.deepcopy(Source._defaults['isochrone']).items():
            kwargs.setdefault(k,v)
        return isochroneFactory(**kwargs)
    
    # Various derived properties
    @property
    def kernel(self):
        return self.models['kernel']

    @property
    def isochrone(self):
        return self.models['isochrone']

    def set_model(self, name, model):
        """ Set a model """
        if not hasattr(self,'models'):
            object.__setattr__(self, 'models',odict())
        self.models[name] = model

    def set_kernel(self,kernel): 
        self.set_model('kernel',kernel)

    def set_isochrone(self,isochrone): 
        self.set_model('isochrone',isochrone)

    def set_params(self,**kwargs):
        """ Set the parameter values """
        for key,value in kwargs.items():
            setattr(self,key,value)

    def get_params(self):
        """ Get an odict of the parameter names and values """
        return odict([(key,param.value) for key,param in self.params.items()])

    def get_free_params(self):
        """ Get an odict of free parameter names and values """
        return odict([(key,param.value) for key,param in self.params.items() if param.free])

    def set_free_params(self, names):
        for name in self.get_params().keys():
            self.setp(name,free=False)
        for name in np.array(names,ndmin=1):
            self.setp(name,free=True)
    
    def get_sync(self,model):
        return self._sync.get(model)

    def reset_sync(self):
        for k in self._sync.keys(): self._sync[k]=False

    def read(self,filename):
        pass

    def write(self,filename,force=False):
        if os.path.exists(filename) and not force:
            raise Exception("Found %s..."%filename)
        out = open(filename,'w')
        out.write(self.dump())
        out.close()

    def set_stellar_mass(self, stellar_mass):
        """ Set the richness to match an input stellar mass. """
        self.richness = stellar_mass/self.isochrone.stellar_mass()

    #@property
    def stellar_mass(self):
        # ADW: I think it makes more sense for this to be
        #return self.richness * self.isochrone.stellarMass()
        return self.isochrone.stellarMass()

    #@property
    def stellar_luminosity(self):
        # ADW: I think it makes more sense for this to be
        #return self.richness * self.isochrone.stellarLuminosity()
        return self.isochrone.stellarLuminosity()

    #@property
    def absolute_magnitude(self, richness):
        #ADW:  Should take self.richness
        return self.isochrone.absolute_magnitude(richness)

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()
