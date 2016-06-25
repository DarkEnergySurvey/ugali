#!/usr/bin/env python
"""
Simple tests of ugali classes
"""

import numpy as np
import scipy

import ugali
import ugali.analysis.isochrone
import ugali.analysis.kernel
import ugali.analysis.source

def test_isochrone():
    iso = ugali.analysis.isochrone.Padova()
    print iso

def test_kernel():
    kernel = ugali.analysis.kernel.Plummer()
    print kernel

def test_source():
    source = ugali.analysis.source.Source()
    print source

def test_factory():
    pass
