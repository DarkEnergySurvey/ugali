#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import numpy as np
import fitsio
from ugali.utils.logger import logger
logger.setLevel(logger.WARN)

CONFIG='ugali/config/config_test.yaml'
LON = RA = 53.92
LAT = DEC = -54.05
IDX = [1,2537,9000]

def test_loglike():
    """ Test ugali.analysis.loglike """
    import ugali.analysis.loglike
    loglike = ugali.analysis.loglike.createLoglike(CONFIG,lon=LON,lat=LAT)

    source = loglike.source
    np.testing.assert_equal(source.richness,1000.)

    # Probability calculations
    np.testing.assert_allclose(loglike.f,0.1945461,rtol=1e-6)
    np.testing.assert_allclose(loglike.u[IDX],
                               [1.11721569e-11, 3.67872840e-03, 3.25216604e-02])
    np.testing.assert_allclose(loglike.b[IDX],
                               [4215.31143651, 9149.29106545, 1698.22182173])
    np.testing.assert_allclose(loglike.p[IDX],
                               [2.65037519e-12, 4.01916371e-04, 1.87905722e-02])
    np.testing.assert_allclose(loglike(),390.153891)
    np.testing.assert_allclose(loglike.ts(),780.30778)
    np.testing.assert_allclose(loglike.nobs,194.546134)

    # Fit the richness
    interval = loglike.richness_interval()
    np.testing.assert_allclose(interval,(1713.4082, 1969.1216))
    lnl,rich,para = loglike.fit_richness()
    np.testing.assert_allclose(lnl,418.666213)
    np.testing.assert_allclose(rich,1839.005237)
    np.testing.assert_allclose(loglike.source.richness,1839.005237)

    # Write membership
    filename = 'test-membership.fits'
    loglike.write_membership(filename)
    mem,hdr = fitsio.read(filename,header=True)

    # Testing output data
    np.testing.assert_allclose(loglike.p,mem['PROB'])
    np.testing.assert_allclose(loglike.catalog.ra,mem['RA'])
    np.testing.assert_allclose(loglike.catalog.dec,mem['DEC'])
    np.testing.assert_allclose(loglike.catalog.color,mem['COLOR'])

    # Testing output metadata
    np.testing.assert_allclose(loglike.ts(),hdr['TS'])
    np.testing.assert_allclose(source.richness,hdr['RICHNESS'])
    np.testing.assert_allclose(source.age,hdr['AGE'])
    np.testing.assert_allclose(source.z,hdr['METALLICITY'])
    np.testing.assert_allclose(source.distance_modulus,hdr['DISTANCE_MODULUS'])
    np.testing.assert_allclose(source.lon,hdr['LON'])
    np.testing.assert_allclose(source.lat,hdr['LAT'])

if __name__ == "__main__":
    import argparse
    description = __doc__
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()

    test_loglike()
