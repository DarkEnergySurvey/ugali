#!/usr/bin/env python

#This is only for backwards compatibility
msg = "'import ugali.analysis.isochrone' is deprecated. "
msg += "Use 'import ugali.isochrone' instead."
DeprecationWarning(msg)

from ugali.isochrone import *
