#!/usr/bin/env python
"""
Simple tests of creating ugali objects
"""

import numpy as np
import scipy

import ugali

def test_isochrone():
    import ugali.isochrone
    iso = ugali.isochrone.Bressan2012()
    print(iso)

    iso2 = ugali.isochrone.factory("Bressan2012")
    assert iso.params == iso2.params

def test_kernel():
    import ugali.analysis.kernel
    kernel = ugali.analysis.kernel.RadialPlummer()
    print(kernel)

    kernel2 = ugali.analysis.kernel.factory("RadialPlummer")
    assert kernel.params == kernel2.params
    
def test_source():
    import ugali.analysis.source
    source = ugali.analysis.source.Source()
    print(source)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    #test_source()
