#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

import numpy as np
import ugali.analysis.results

def test_surface_brightness():
    
    # Draco values from McConnachie et al. 2012
    abs_mag = -8.8
    app_mag = 10.6
    distance = 76 #kpc
    distance_modulus = 19.40
    a_half_arcmin = 10.0 # arcmin
    a_physical_kpc = 0.221 # kpc
    ellip = 0.31
    mu0 = 26.1 # mag/arcsec^2

    # Convert to azimuthally average
    r_half_arcmin = a_half_arcmin * np.sqrt(1-ellip)
    r_physical_kpc = a_physical_kpc * np.sqrt(1-ellip)

    mu1 = ugali.analysis.results.surfaceBrightness(abs_mag, r_physical_kpc, distance)
    mu2 = ugali.analysis.results.surfaceBrightness2(app_mag, r_half_arcmin)

    np.testing.assert_allclose(mu1,mu0,atol=2e-2)
    np.testing.assert_allclose(mu2,mu0,atol=2e-2)
