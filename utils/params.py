#!/usr/bin/env python

class Parameter(object):
    """
    Parameter class for storing value and bounds.

    Adapted from MutableNum from https://gist.github.com/jheiv/6656349
    """
    __value__ = None
    __bounds__ = None
    def __init__(self, value, bounds=None): self.set(value,bounds)

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
    def __repr__(self):         return "%s(%s)" % (self.__class__.__name__, self.__value__)
    # Return the type of the inner value
    def innertype(self):        return type(self.__value__)

    @property
    def bounds(self):
        return self.__bounds__

    @property
    def value(self):
        return self.__value__

    def set_bounds(self, bounds):
        if bounds is None: return
        else: self.__bounds__ = bounds

    def check_bounds(self, value):
        if self.__bounds__ is None:
            return
        if not (self.__bounds__[0] <= value <= self.__bounds__[1]):
            msg="Value outside bounds: %.2g [%.2g,%.2g]"
            msg=msg%(value,self.__bounds__[0],self.__bounds__[1])
            raise ValueError(msg)

    def set_value(self, value):
        self.check_bounds(value)
        if   isinstance(value, (int, long, float)): self.__value__ = value
        elif isinstance(value, self.__class__): self.__value__ = value.__value__
        else: raise TypeError("Numeric type required")

    def set(self, value, bounds=None):
        self.set_bounds(bounds)
        self.set_value(value)


if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args
