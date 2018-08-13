#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import unittest
# Execute tests in order: https://stackoverflow.com/a/22317851/4075339
unittest.TestLoader.sortTestMethodsUsing = None

import numpy as np
import fitsio

import ugali.analysis.loglike
from ugali.utils.logger import logger
logger.setLevel(logger.WARN)

CONFIG='tests/config.yaml'
LON = RA = 53.92
LAT = DEC = -54.05
IDX = [1,2537,9000]

class TestLoglike(unittest.TestCase):
    """Test the loglikelihood"""

    def setUp(self):
        self.loglike = ugali.analysis.loglike.createLoglike(CONFIG,lon=LON,lat=LAT)
        self.source = self.loglike.source
        self.filename = 'test-membership.fits'

    def test_initial_config(self):
        # Likelihood configuration.
        np.testing.assert_equal(self.source.isochrone.name,'Bressan2012')
        np.testing.assert_equal(self.source.kernel.name,'RadialPlummer')
        np.testing.assert_equal(self.source.richness,1000.)

    def test_initial_probability(self):
        # Probability calculations

        np.testing.assert_allclose(self.loglike.f,0.08614111,rtol=1e-6)
     
        np.testing.assert_allclose(self.loglike.u[IDX],
                                   [5.29605173e-03, 1.80040569e-03, 5.52283081e-09],
                                   rtol=1e-6)
        np.testing.assert_allclose(self.loglike.b[IDX],
                                   [4215.31143651, 9149.29106545, 1698.22182173],
                                   rtol=1e-6)
        np.testing.assert_allclose(self.loglike.p[IDX],
                                   [1.25480793e-03, 1.96742181e-04, 3.25212568e-09],
                                   rtol=1e-6)
        np.testing.assert_allclose(self.loglike(),3947.9703876)
        np.testing.assert_allclose(self.loglike.ts(),7895.94077)
        np.testing.assert_allclose(self.loglike.nobs,86.1411187)

    def test_fit_richness(self):
        # Fit the richness

        interval = self.loglike.richness_interval()
        np.testing.assert_allclose(interval,(31516.82584, 32836.37888))
                                             
        lnl,rich,para = self.loglike.fit_richness()
        np.testing.assert_allclose(lnl,8443.79621)
        np.testing.assert_allclose(rich,32171.788052)
        np.testing.assert_allclose(self.loglike.source.richness,32171.78805)

    def test_write_membership(self):
        # Write membership
        self.loglike.write_membership(self.filename)

    def test_output(self):
        # Test output data and metadata
        mem,hdr = fitsio.read(self.filename,header=True)

        np.testing.assert_allclose(self.loglike.p,mem['PROB'])
        np.testing.assert_allclose(self.loglike.catalog.ra,mem['RA'])
        np.testing.assert_allclose(self.loglike.catalog.dec,mem['DEC'])
        np.testing.assert_allclose(self.loglike.catalog.color,mem['COLOR'])
 
        # Testing output metadata
        np.testing.assert_allclose(self.loglike.ts(),hdr['TS'])
        np.testing.assert_allclose(self.source.richness,hdr['RICHNESS'])
        np.testing.assert_allclose(self.source.age,hdr['AGE'])
        np.testing.assert_allclose(self.source.z,hdr['METALLICITY'])
        np.testing.assert_allclose(self.source.distance_modulus,hdr['DISTANCE_MODULUS'])
        np.testing.assert_allclose(self.source.lon,hdr['LON'])
        np.testing.assert_allclose(self.source.lat,hdr['LAT'])

        
#def test_loglike():
#    """ Test ugali.analysis.loglike """
#    import ugali.analysis.loglike
#    loglike = ugali.analysis.loglike.createLoglike(CONFIG,lon=LON,lat=LAT)
# 
#    source = loglike.source
#    np.testing.assert_equal(source.isochrone.name,'Bressan2012')
#    np.testing.assert_equal(source.kernel.name,'RadialPlummer')
#    np.testing.assert_equal(source.richness,1000.)
# 
#    # Probability calculations
#    np.testing.assert_allclose(loglike.f,0.08614111,rtol=1e-6)
# 
#    np.testing.assert_allclose(loglike.u[IDX],
#                               [5.29605173e-03, 1.80040569e-03, 5.52283081e-09],
#                               rtol=1e-6)
#    np.testing.assert_allclose(loglike.b[IDX],
#                               [4215.31143651, 9149.29106545, 1698.22182173],
#                               rtol=1e-6)
#    np.testing.assert_allclose(loglike.p[IDX],
#                               [1.25480793e-03, 1.96742181e-04, 3.25212568e-09],
#                               rtol=1e-6)
#    np.testing.assert_allclose(loglike(),3947.9703876)
#    np.testing.assert_allclose(loglike.ts(),7895.94077)
#    np.testing.assert_allclose(loglike.nobs,86.1411187)
# 
#    # Fit the richness
#    interval = loglike.richness_interval()
#    np.testing.assert_allclose(interval,(31516.82584, 32836.37888))
#                                        
#    lnl,rich,para = loglike.fit_richness()
#    np.testing.assert_allclose(lnl,8443.79621)
#    np.testing.assert_allclose(rich,32171.788052)
#    np.testing.assert_allclose(loglike.source.richness,32171.78805)
# 
#    # Write membership
#    filename = 'test-membership.fits'
#    loglike.write_membership(filename)
#    mem,hdr = fitsio.read(filename,header=True)
# 
#    # Testing output data
#    np.testing.assert_allclose(loglike.p,mem['PROB'])
#    np.testing.assert_allclose(loglike.catalog.ra,mem['RA'])
#    np.testing.assert_allclose(loglike.catalog.dec,mem['DEC'])
#    np.testing.assert_allclose(loglike.catalog.color,mem['COLOR'])
# 
#    # Testing output metadata
#    np.testing.assert_allclose(loglike.ts(),hdr['TS'])
#    np.testing.assert_allclose(source.richness,hdr['RICHNESS'])
#    np.testing.assert_allclose(source.age,hdr['AGE'])
#    np.testing.assert_allclose(source.z,hdr['METALLICITY'])
#    np.testing.assert_allclose(source.distance_modulus,hdr['DISTANCE_MODULUS'])
#    np.testing.assert_allclose(source.lon,hdr['LON'])
#    np.testing.assert_allclose(source.lat,hdr['LAT'])

if __name__ == "__main__":
    unittest.main()

    #test_loglike()
