#!/usr/bin/env python
import yaml
from collections import OrderedDict as odict
import numpy as np
import copy

from ugali.analysis.kernel import kernelFactory
from ugali.analysis.isochrone2 import isochroneFactory
from ugali.analysis.model import Model,Parameter

class Richness(Model):
    """
    Dummy model to hold the richness, which is not
    directly connected to either the spatial or 
    color information and doesn't require a sync
    when updated.
    """
    _params = odict([
        ('richness', Parameter(1.0, [0.0,  np.inf])),
    ])

# Just to be consistent
def richnessFactory(name='Richness',**kwargs):
    return Richness()

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

    def __init__(self,name=None,srcmdl=None):
        self._load(srcmdl)
        self.name = name

        # Toggle for tracking which models need to be synched
        self._sync = odict([(k,True) for k in self.models.keys()])

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
        for key,model in self.models.items():
            try:
                return model.getp(name).value
            except KeyError:
                continue
        # Raises AttributeError
        return object.__getattribute__(self,name)
     
    def __setattr__(self, name, value):
        #for key,model in self.models.items():
        #    if name in model.params:
        #        self._sync[key] = True
        #        return setattr(model, name, value)
        for key,model in self.models.items():
            try:
                ret = model.setp(name, value)
                self._sync[key] = True
                return ret
            except KeyError:
                continue
        # Raises AttributeError?
        return object.__setattr__(self, name, value)

    def _load(self,srcmdl=None):
        if srcmdl is None: 
            params = dict()
        elif isinstance(srcmdl,basestring): 
            params = yaml.load(open(srcmdl))
        else:
            params = copy.deepcopy(srcmdl)
        for k,v in copy.deepcopy(self._defaults).items():
            params.setdefault(k,v)
            
        richness  = self.createRichness(**params['richness'])
        kernel    = self.createKernel(**params['kernel'])
        isochrone = self.createIsochrone(**params['isochrone'])

        self.set_model('richness',richness)
        self.set_model('spatial',kernel)
        self.set_model('color',isochrone)
    
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
        return self.models['spatial']

    @property
    def isochrone(self):
        return self.models['color']

    @property
    def params(self):
        params = odict([])
        for key,model in self.models.items():
            params.update(model.params)
        return params

    def set_model(self, name, model):
        if not hasattr(self,'models'):
            object.__setattr__(self, 'models',odict())
        self.models[name] = model

    def set_kernel(self,kernel): self.set_model('spatial',kernel)
    def set_isochrone(self,isochrone): self.set_model('color',isochrone)

    def set_params(self,**kwargs):
        for key,value in kwargs.items():
            setattr(self,key,value)

    def getp(self,name):
        return self.params['name']

    def setp(self,name,*args,**kwargs):
        self.params['name'].setp(*args,**kwargs)
    
    def get_sync(self,model):
        return self._sync.get(model)

    def reset_sync(self):
        for k in self._sync.keys(): self._sync[k]=False

    def read(self,filename):
        pass

    def write(self,filename):
        pass

    def stellar_mass(self):
        # ADW: I think it makes more sense for this to be
        #return self.richness * self.isochrone.stellarMass()
        return self.isochrone.stellarMass()

    def stellar_luminosity(self):
        # ADW: I think it makes more sense for this to be
        #return self.richness * self.isochrone.stellarLuminosity()
        return self.isochrone.stellarLuminosity()

    def absolute_magnitude(self, richness):
        return self.isochrone.absolute_magnitude(richness)

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    opts = parser.parse_args()
