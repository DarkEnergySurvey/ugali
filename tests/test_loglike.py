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
    np.testing.assert_equal(source.isochrone.name,'Bressan2012')
    np.testing.assert_equal(source.kernel.name,'RadialPlummer')
    np.testing.assert_equal(source.richness,1000.)

    # Probability calculations
    np.testing.assert_allclose(loglike.f,0.08614111,rtol=1e-6)

    np.testing.assert_allclose(loglike.u[IDX],
                               [5.29605173e-03, 1.80040569e-03, 5.52283081e-09],
                               rtol=1e-6)
    np.testing.assert_allclose(loglike.b[IDX],
                               [4215.31143651, 9149.29106545, 1698.22182173])
    np.testing.assert_allclose(loglike.p[IDX],
                               [1.25480793e-03, 1.96742181e-04, 3.25212568e-09])
    np.testing.assert_allclose(loglike(),3947.9703876)
    np.testing.assert_allclose(loglike.ts(),7895.94077)
    np.testing.assert_allclose(loglike.nobs,86.1411187)

    # Fit the richness
    interval = loglike.richness_interval()
    np.testing.assert_allclose(interval,(31516.82584, 32836.37888))
                                        
    lnl,rich,para = loglike.fit_richness()
    np.testing.assert_allclose(lnl,8443.79621)
    np.testing.assert_allclose(rich,32171.788052)
    np.testing.assert_allclose(loglike.source.richness,32171.78805)

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
