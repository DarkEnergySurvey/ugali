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

        np.testing.assert_allclose(self.loglike.f,0.08595560,rtol=1e-6)
        np.testing.assert_allclose(self.loglike.u[IDX],
                                   [5.29605173e-03, 1.80040569e-03, 5.52283081e-09],
                                   rtol=1e-6)
        np.testing.assert_allclose(self.loglike.b[IDX],
                                   [4215.31143651, 9149.29106545, 1698.22182173],
                                   rtol=1e-6)
        np.testing.assert_allclose(self.loglike.p[IDX],
                                   [1.25480793e-03, 1.96742181e-04, 3.25212568e-09],
                                   rtol=1e-6)

        np.testing.assert_allclose(self.loglike(),3948.1559048)
        np.testing.assert_allclose(self.loglike.ts(),7896.31181)
        np.testing.assert_allclose(self.loglike.nobs,85.9556015)

    def test_fit_richness(self):
        # Fit the richness
        interval = self.loglike.richness_interval()
        np.testing.assert_allclose(interval,(31596.21551, 32918.707276))
                                             
        lnl,rich,para = self.loglike.fit_richness()
        np.testing.assert_allclose(lnl,8449.77225)
        np.testing.assert_allclose(rich,32252.807226)
        np.testing.assert_allclose(self.loglike.source.richness,rich)

    def test_write_membership(self):
        # Write membership
        self.loglike.write_membership(self.filename)

        # Read membership and metadata
        mem,hdr = fitsio.read(self.filename,header=True)

        np.testing.assert_allclose(self.loglike.p,mem['PROB'])
        np.testing.assert_allclose(self.loglike.catalog.ra,mem['RA'])
        np.testing.assert_allclose(self.loglike.catalog.dec,mem['DEC'])
        np.testing.assert_allclose(self.loglike.catalog.color,mem['COLOR'])
 
        # Testing output metadata
        print (self.loglike.ts(),hdr['TS'])
        np.testing.assert_allclose(self.loglike.ts(),hdr['TS'])
        np.testing.assert_allclose(self.source.richness,hdr['RICHNESS'])
        np.testing.assert_allclose(self.source.age,hdr['AGE'])
        np.testing.assert_allclose(self.source.z,hdr['METALLICITY'])
        np.testing.assert_allclose(self.source.distance_modulus,hdr['DISTANCE_MODULUS'])
        np.testing.assert_allclose(self.source.lon,hdr['LON'])
        np.testing.assert_allclose(self.source.lat,hdr['LAT'])

if __name__ == "__main__":
    unittest.main()
