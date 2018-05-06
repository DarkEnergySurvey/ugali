#!/usr/bin/env python
"""
Simple tests of ugali classes
"""

import numpy as np
import scipy

import ugali

def test_isochrone():
    import ugali.isochrone
    iso = ugali.isochrone.Padova()
    print(iso)

    iso2 = ugali.isochrone.factory("Padova")
    assert iso.params == iso2.params

def test_kernel():
    import ugali.analysis.kernel
    kernel = ugali.analysis.kernel.Plummer()
    print(kernel)

    kernel2 = ugali.analysis.kernel.factory("Plummer")
    assert kernel.params == kernel2.params
    
def test_source():
    import ugali.analysis.source
    source = ugali.analysis.source.Source()
    print(source)
