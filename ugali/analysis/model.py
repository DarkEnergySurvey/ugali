#!/usr/bin/env python
"""
A Model object is just a container for a set of Parameter.
Implements __getattr__ and __setattr__.

The model has a set of default parameters stored in Model._params. 
Careful, if these are changed, they will be changed for all 
subsequent instances of the Model.

The parameters for a given instance of a model are stored in the
Model.params attribute. This attribute is a deepcopy of 
Model._params created during instantiation.

"""
from collections import OrderedDict as odict
import numpy as np
import copy
import yaml

def indent(string,width=0): 
    return '{0:>{1}}{2}'.format('',width,string)

def asscalar(a):
    """ https://github.com/numpy/numpy/issues/4701 """
    # Do we want to check that the value is numeric?
    #if   isinstance(value, (int, long, float)): return value
    try:
        return np.asscalar(a)
    except AttributeError, e:
        return np.asscalar(np.asarray(a))


class Model(object):
    # The _params member is an ordered dictionary
    # of Parameter objects. (Should be immutable tuple.)
    _params = odict([])
    # The _mapping is an alternative name mapping
    # for the parameters in _params
    _mapping = odict([])

    def __init__(self,*args,**kwargs):
        self.name = self.__class__.__name__
        self.params = copy.deepcopy(self._params)
        self.set_attributes(**kwargs)
        #pars = dict(**kwargs)
        #for name, value in pars.items():
        #    # Raise AttributeError if attribute not found
        #    self.__getattr__(name) 
        #    # Set attribute
        #    self.__setattr__(name,value)
        
        # In case no properties were set, cache anyway
        self._cache()

    def __getattr__(self,name):
        # Return 'value' of parameters
        # __getattr__ tries the usual places first.
        #if name in self._mapping:
        #    return self.__getattr__(self._mapping[name])
        #if name in self._params:
        #    return self.getp(name).value
        #else:
        #    # Raises AttributeError
        #    return object.__getattribute__(self,name)
        if name in self._params or name in self._mapping:
            return self.getp(name).value
        else:
            # Raises AttributeError
            return object.__getattribute__(self,name)

    def __setattr__(self, name, value):
        ## Call 'set_value' on parameters
        ## __setattr__ tries the usual places first.
        #if name in self._mapping.keys():
        #    return self.__setattr__(self._mapping[name],value)
        #if name in self._params:
        #    self.setp(name, value)
        #else:
        #    return object.__setattr__(self, name, value)
        if name in self._params or name in self._mapping:
            self.setp(name, value)
        else:
            # Why is this a return statement
            return object.__setattr__(self, name, value)

    def __str__(self,indent=0):
        ret = '{0:>{2}}{1}'.format('',self.name,indent)
        if len(self.params)==0:
            pass
        else:            
            ret += '\n{0:>{2}}{1}'.format('','Parameters:',indent+2)
            width = len(max(self.params.keys(),key=len))
            for name,value in self.params.items():
                par = '{0!s:{width}} : {1!r}'.format(name,value,width=width)
                ret += '\n{0:>{2}}{1}'.format('',par,indent+4)
        return ret

    def getp(self, name):
        """ 
        Get the named parameter.

        Parameters
        ----------
        name : string
            The parameter name.

        Returns
        -------
        param : 
            The parameter object.
        """
        name = self._mapping.get(name,name)
        return self.params[name]

    def setp(self, name, value=None, bounds=None, free=None, errors=None):
        """ 
        Set the value (and bounds) of the named parameter.

        Parameters
        ----------
        name : string
            The parameter name.
        value: 
            The value of the parameter
        bounds: None
            The bounds on the parameter
        Returns
        -------
        None
        """
        name = self._mapping.get(name,name)
        self.params[name].set(value,bounds,free,errors)
        self._cache(name)

    def set_attributes(self, **kwargs):
        """
        Set a group of attributes (parameters and members).  Calls
        `setp` directly, so kwargs can include more than just the
        parameter value (e.g., bounds, free, etc.).
        """
        kwargs = dict(kwargs)
        for name,value in kwargs.items():
            # Raise AttributeError if param not found
            self.__getattr__(name) 
            # Set attributes
            try: self.setp(name,**value)
            except TypeError:
                try:  self.setp(name,*value)
                except (TypeError,KeyError):  
                    self.__setattr__(name,value)
        
    def todict(self):
        ret = odict(name = self.__class__.__name__)
        ret.update(self.params)
        return ret

    def dump(self):
        return yaml.dump(self.todict())

    def _cache(self, name=None):
        """ 
        Method called in _setp to cache any computationally
        intensive properties after updating the parameters.

        Parameters
        ----------
        name : string
           The parameter name.

        Returns
        -------
        None
        """
        pass

    #@property
    #def params(self):
    #    return self._params


class Parameter(object):
    """
    Parameter class for storing a value, bounds, and freedom.

    Adapted from `MutableNum` from https://gist.github.com/jheiv/6656349
    """
    __value__ = None
    __bounds__ = None
    __free__ = False
    __errors__ = None

    def __init__(self, value, bounds=None, free=None, errors=None): 
        self.set(value,bounds,free,errors)

    # Comparison Methods
    def __eq__(self, x):        return self.__value__ == x
    def __ne__(self, x):        return self.__value__ != x
    def __lt__(self, x):        return self.__value__ <  x
    def __gt__(self, x):        return self.__value__ >  x
    def __le__(self, x):        return self.__value__ <= x
    def __ge__(self, x):        return self.__value__ >= x
    def __cmp__(self, x):       return 0 if self.__value__ == x else 1 if self.__value__ > 0 else -1
    # Unary Ops
    def __pos__(self):          return +self.__value__
    def __neg__(self):          return -self.__value__
    def __abs__(self):          return abs(self.__value__)
    # Bitwise Unary Ops
    def __invert__(self):       return ~self.__value__
    # Arithmetic Binary Ops
    def __add__(self, x):       return self.__value__ + x
    def __sub__(self, x):       return self.__value__ - x
    def __mul__(self, x):       return self.__value__ * x
    def __div__(self, x):       return self.__value__ / x
    def __mod__(self, x):       return self.__value__ % x
    def __pow__(self, x):       return self.__value__ ** x
    def __floordiv__(self, x):  return self.__value__ // x
    def __divmod__(self, x):    return divmod(self.__value__, x)
    def __truediv__(self, x):   return self.__value__.__truediv__(x)
    # Reflected Arithmetic Binary Ops
    def __radd__(self, x):      return x + self.__value__
    def __rsub__(self, x):      return x - self.__value__
    def __rmul__(self, x):      return x * self.__value__
    def __rdiv__(self, x):      return x / self.__value__
    def __rmod__(self, x):      return x % self.__value__
    def __rpow__(self, x):      return x ** self.__value__
    def __rfloordiv__(self, x): return x // self.__value__
    def __rdivmod__(self, x):   return divmod(x, self.__value__)
    def __rtruediv__(self, x):  return x.__truediv__(self.__value__)
    # Bitwise Binary Ops
    def __and__(self, x):       return self.__value__ & x
    def __or__(self, x):        return self.__value__ | x
    def __xor__(self, x):       return self.__value__ ^ x
    def __lshift__(self, x):    return self.__value__ << x
    def __rshift__(self, x):    return self.__value__ >> x
    # Reflected Bitwise Binary Ops
    def __rand__(self, x):      return x & self.__value__
    def __ror__(self, x):       return x | self.__value__
    def __rxor__(self, x):      return x ^ self.__value__
    def __rlshift__(self, x):   return x << self.__value__
    def __rrshift__(self, x):   return x >> self.__value__
    # ADW: Don't allow compound assignments
    ## Compound Assignment
    #def __iadd__(self, x):      self.set(self + x); return self
    #def __isub__(self, x):      self.set(self - x); return self
    #def __imul__(self, x):      self.set(self * x); return self
    #def __idiv__(self, x):      self.set(self / x); return self
    #def __imod__(self, x):      self.set(self % x); return self
    #def __ipow__(self, x):      self.set(self **x); return self
    # Casts
    def __nonzero__(self):      return self.__value__ != 0
    def __int__(self):          return self.__value__.__int__()    
    def __float__(self):        return self.__value__.__float__()  
    def __long__(self):         return self.__value__.__long__()   
    # Conversions
    def __oct__(self):          return self.__value__.__oct__()    
    def __hex__(self):          return self.__value__.__hex__()    
    def __str__(self):          return self.__value__.__str__()    
    # Random Ops
    def __index__(self):        return self.__value__.__index__()  
    def __trunc__(self):        return self.__value__.__trunc__()  
    def __coerce__(self, x):    return self.__value__.__coerce__(x)
    # Represenation
    # ADW: This should probably be __str__ not __repr__
    def __repr__(self):         
        if self.bounds:
            # if bounds is None this breaks
            return "%s(%s, [%s, %s], %s)"%(self.__class__.__name__, self.value,self.bounds[0],self.bounds[1],self.free)
        else:
            return "%s(%s, [%s, %s], %s)"%(self.__class__.__name__, self.value,self.bounds,self.bounds,self.free)

    # Return the type of the inner value
    def innertype(self):  return type(self.__value__)

    @property
    def bounds(self):
        return self.__bounds__

    @property
    def value(self):
        return self.__value__

    @property
    def free(self):
        return self.__free__

    @property
    def errors(self):
        return self.__errors__

    def item(self): 
        """ For asscalar """
        return self.value

    def check_bounds(self, value):
        if self.__bounds__ is None:
            return
        if not (self.__bounds__[0] <= value <= self.__bounds__[1]):
            msg="Value outside bounds: %.2g [%.2g,%.2g]"
            msg=msg%(value,self.__bounds__[0],self.__bounds__[1])
            raise ValueError(msg)

    def set_bounds(self, bounds):
        if bounds is None: return
        self.__bounds__ = [asscalar(b) for b in bounds]

    def set_value(self, value):
        if value is None: return
        self.check_bounds(value)
        self.__value__ = asscalar(value)

    def set_free(self, free):
        if free is None: return
        else: self.__free__ = bool(free)

    def set_errors(self, errors):
        if errors is None: return
        self.__errors__ = [asscalar(e) for e in errors]

    def set(self, value=None, bounds=None, free=None, errors=None):
        # Probably want to reset bounds if set fails
        self.set_bounds(bounds)
        self.set_value(value)
        self.set_free(free)
        self.set_errors(errors)

    def todict(self):
        return odict(value=self.value,bounds=self.bounds,
                     free=self.free,errors=self.errors)

    def dump(self):
        return yaml.dump(self)

    @staticmethod
    def representer(dumper, data):
        """ 
        http://stackoverflow.com/a/14001707/4075339
        http://stackoverflow.com/a/21912744/4075339 
        """
        tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
        return dumper.represent_mapping(tag,data.todict().items(),flow_style=True)

def odict_representer(dumper, data):
    """ http://stackoverflow.com/a/21912744/4075339 """
    # Probably belongs in a util
    return dumper.represent_mapping(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,data.items())

yaml.add_representer(odict,odict_representer)
yaml.add_representer(Parameter,Parameter.representer)

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args
