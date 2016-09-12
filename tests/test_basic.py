#!/usr/bin/env python
"""
Simple tests of ugali classes
"""

import numpy as np
import scipy

import ugali

def test_isochrone():
    import ugali.analysis.isochrone
    iso = ugali.analysis.isochrone.Padova()
    print iso

def test_kernel():
    import ugali.analysis.kernel
    kernel = ugali.analysis.kernel.Plummer()
    print kernel

def test_source():
    import ugali.analysis.source
    source = ugali.analysis.source.Source()
    print source

def test_factory():
    import ugali.analysis.isochrone as isochrone
    import ugali.analysis.kernel as kernel
    pass
