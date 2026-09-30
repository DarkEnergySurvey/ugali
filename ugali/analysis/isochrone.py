#!/usr/bin/env python

#This is only for backwards compatibility
msg = "'import ugali.analysis.isochrone' is deprecated. "
msg += "Use 'import ugali.isochrone' instead."
import warnings
warnings.warn(msg, DeprecationWarning, stacklevel=2)

from ugali.isochrone import *
